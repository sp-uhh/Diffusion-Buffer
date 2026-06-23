import time
from math import ceil
import warnings

import numpy as np
import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage

import torchaudio

import torchaudio
import wandb

from sgmse import sampling
from sgmse.sdes import SDERegistry
from sgmse.backbones import BackboneRegistry
from sgmse.util.inference import evaluate_model
from sgmse.util.other import pad_spec
from sgmse.util.graphics import visualize_example
from sgmse.util.agc import normalize_online
from utils import MultiResolutionSTFTLoss

VIS_EPOCHS = 10

class ScoreModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum time (3e-2 by default)")
        parser.add_argument("--num_eval_files", type=int, default=20, help="Number of files for speech enhancement performance evaluation during training.")
        parser.add_argument("--loss_type", type=str, default="mse", help="The type of loss function to use.")
        parser.add_argument("--loss_abs_exponent", type=float, default= 0.5,  help="magnitude transformation in the loss term")
        parser.add_argument("--output_scale", type=str, choices=('sigma', 'time', 'no'), default= 'time',  help="backbone model scale before last output layers")
        parser.add_argument("--timestep_type_inf", type=str, default= 'default',  help="linear")
        parser.add_argument("--audiologs_every_epoch", type=int, default=1, help="log audios every nth epoch")
        parser.add_argument("--speclogs_every_epoch", type=int, default=1, help="log specs every nth epoch")
        parser.add_argument("--diff_buffer_range", type=int, nargs=2, required=False, default=[16,17],help="Diffusion buffer length B is sampled from a uniform distribution between [min, max For example [16, 17] means B=16.")
        parser.add_argument("--diff_buffer_length_inference", type=int, default=15, help="Diff buffer Length")
        parser.add_argument("--fixed_buffer_steps",   type=str, default='rdm', help="how to sample the increasing sequences of diffusion-timesteps")
        parser.add_argument("--compile_model", action="store_true", help="Compile the model using torch.compile()")
        parser.add_argument("--non_compile_model", action="store_false", dest='compile_model', help="Compile the model using torch.compile()")
        parser.set_defaults(compile_model=True)

        return parser

    def __init__(
        self, backbone, sde, lr=1e-4, ema_decay=0.999, t_eps=3e-2, loss_abs_exponent=0.5, diff_buffer_length_inference=15,
        num_eval_files=20, loss_type='mse', data_module_cls=None, output_scale='time', fixed_buffer_steps=True, diff_buffer_range=[5,35],
        timestep_type_inf = 'linear', audiologs_every_epoch = 0, speclogs_every_epoch = 0, compile_model=True,  **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model.
            sde: The SDE that defines the diffusion process.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
        """
        super().__init__()
        # Initialize Backbone DNN
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)
        if compile_model:
            self.dnn = torch.compile(self.dnn, dynamic=False)
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files
        self.loss_abs_exponent = loss_abs_exponent
        self.output_scale = output_scale
        self.N_inf = diff_buffer_length_inference
        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
        self.timestep_type_inf = timestep_type_inf
        self.audiologs_every_epoch = audiologs_every_epoch
        self.speclogs_every_epoch = speclogs_every_epoch
        if self.loss_type == 'gencon_mrstft':
            self.mrstftloss = MultiResolutionSTFTLoss(factor_sc=.5, factor_mag=.5).to(self.device) #factor para taken from demucs
        if self.loss_type == 'gencon_mel':
            self.melloss  = torchaudio.transforms.MelSpectrogram(kwargs['fs'])
        self.fixed_buffer_steps = fixed_buffer_steps
        self.diff_buffer_range = diff_buffer_range
        self.num_frames = self.data_module.num_frames

        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    
    def _loss(self, score, sigmas, z, gt=None):    

        e = self.loss_abs_exponent

        if self.loss_type == 'mse':
            err = sigmas*score + z 
            losses = torch.square(err.abs()) #denoising score matching loss
        elif self.loss_type == 'onestep':  #also known as data prediction loss
            if gt is None:
                raise ValueError("gt must be provided for onestep loss")
            #l2 between score and gt
            losses = torch.square(torch.abs(score - gt))

        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss


    def _step(self, batch, batch_idx):
        x, y = batch


        buffer_length = torch.randint(self.diff_buffer_range[0], self.diff_buffer_range[1], (1,))
        
        #construct buffer
        perturbed_data, std, z, time_embeddings = self.sde.diff_buffer_matrices(buffer_length, x, y, 
                                                    self.t_eps, self.fixed_buffer_steps)


        
        y_condion = None
        #scale shape [BS, 1, Freq, NumFrames]
        if self.output_scale=='time':
            scale = time_embeddings[:,None, None, :]
            scale[:,:,:,:-buffer_length] = 1
        elif self.output_scale=='sigma':
            #not supported atm
            pass
            scale = std
        elif self.output_scale=='no':
            scale = torch.ones_like(std)
        else:
            raise ValueError('output scale not implemented!')
        
        nn_out = self(perturbed_data, time_embeddings, y, scale) #shape (BS, 1, Freq, NumFrames) complex-valued
        #crop frames only to the buffer
        if type(nn_out) == tuple:
            xt_generator, y_condion = nn_out
            nn_out = xt_generator
            y_condion = y_condion[:,:,:, -buffer_length: ]
            
        nn_out = nn_out[:,:, :, -buffer_length: ]
        z = z[:,:,:, -buffer_length: ]
        std = std[:,:,:, -buffer_length: ]
        x = x[:,:,:, -buffer_length: ]       
        residual_noise = y[:,:,:, -buffer_length: ] - x
        loss = self._loss(nn_out, std, z, gt=x, environmental=residual_noise, y_condion = y_condion)
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)
        
        if self.num_eval_files != 0 and batch_idx==0:
            basic_metrics, _, spec, audio = evaluate_model(self, self.num_eval_files, 
                                                           self.N_inf, 
                                                        num_frames=self.num_frames, fixed_buffer_steps=self.fixed_buffer_steps)
            for key, value in basic_metrics.items():
                self.log(key, np.mean(value), on_step=False, on_epoch=True)

            if self.audiologs_every_epoch > 0:
                if self.current_epoch % self.audiologs_every_epoch == 0:
                    self._log_audio(audio)
            if self.speclogs_every_epoch > 0:
                if self.current_epoch % self.speclogs_every_epoch == 0:
                    self._log_spec(spec)
        return loss


    def _log_audio(self, audio):
        if audio is not None:
            sr = self.data_module.fs
            for idx, (y, x_hat, x) in enumerate(zip(audio["y"], audio["x_hat"], audio["x"])):
                if type(self.logger).__name__ == "TensorBoardLogger":
                    if self.current_epoch == 0:
                        self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Mix/{idx}", (y * .9/np.max(np.abs(y)))[..., np.newaxis], sample_rate=sr, global_step=None)
                        self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Clean/{idx}", (x * .9/np.max(x))[..., np.newaxis], sample_rate=sr, global_step=None)
                    self.logger.experiment.add_audio(f"Epoch={self.current_epoch} Estimate/{idx}", (x_hat * .9/np.max(np.abs(x_hat)))[..., np.newaxis], sample_rate=sr, global_step=None)
                elif type(self.logger).__name__ == "WandbLogger":
                    if self.current_epoch == 0:
                        self.logger.experiment.log({ f"Audio/Epoch={self.current_epoch}/{idx}": wandb.Audio((y * .9/np.max(np.abs(y)))[..., np.newaxis], caption="Mixture", sample_rate=sr) })
                        self.logger.experiment.log({ f"Audio/Epoch={self.current_epoch}/{idx}": wandb.Audio((x * .9/np.max(np.abs(x)))[..., np.newaxis], caption="Clean", sample_rate=sr) })
                    self.logger.experiment.log({ f"Audio/Epoch={self.current_epoch}/{idx}": wandb.Audio((x_hat * .9/np.max(np.abs(x_hat)))[..., np.newaxis], caption="Estimate", sample_rate=sr) })

                           
    def _log_spec(self, spec):
        if spec is not None:
            figures = []
            for idx, (y_stft, x_hat_stft, x_stft) in enumerate(zip(spec["y"], spec["x_hat"], spec["x"])):
                figures.append(
                    visualize_example(
                    torch.abs(y_stft), 
                    torch.abs(x_hat_stft), 
                    torch.abs(x_stft), 
                    sample_rate=self.data_module.fs,
                    hop_len=self.data_module.hop_length,
                    return_fig=True)
                    )
            if type(self.logger).__name__ == "TensorBoardLogger":
                self.logger.experiment.add_figure(f"Epoch={self.current_epoch}", figures, close=True, global_step=None)
            elif type(self.logger).__name__ == "WandbLogger":
                self.logger.experiment.log({f"Spec/Epoch={self.current_epoch}": [wandb.Image(fig, caption=f"Sample {i}") for (i, fig) in enumerate(figures)]})



    def forward(self, x, t, y, scale=None):
        args = [x, t, y, scale]
        score = self.dnn(*args)
        return score

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(self, predictor_name, corrector_name, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn
        
    def get_cached_pc_sampler(self, predictor_name, corrector_name, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_cached_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_cached_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn



    def get_ode_sampler(self, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_ode_sampler(sde, self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return sample, ns
            return batched_sampling_fn

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)




    def enhance(self, y, sampler_type="pc", predictor="reverse_diffusion",
        corrector="ald", N=30, corrector_steps=1, snr=0.5, timeit=False, 
        do_normalize_online=False, vad_threshold=1e-3, 
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        start = time.time()
        T_orig = y.size(1)
        if not do_normalize_online:
            if self.data_module.normalize == "noisy":
                norm_factor_y, norm_factor_x = y.abs().max(), y.abs().max()
            elif self.data_module.normalize in ["noisy_only", "separate"]:
                norm_factor_y, norm_factor_x = y.abs().max(), 1.
            elif self.data_module.normalize == "not":
                norm_factor_y, norm_factor_x = 1., 1.
            y = y / norm_factor_y

        Y = torch.unsqueeze(self._stft(y.to(self.device)), 0)
        Y[..., : self.data_module.remove_bins, :] = 0. + 1j*0.
        Y = pad_spec(Y)
        num_frames = Y.shape[3]

        if do_normalize_online:
            Y = normalize_online(Y.squeeze(), fs=self.data_module.fs, stft_kwargs=self.data_module.stft_kwargs, vad_threshold=vad_threshold)
            Y = Y.unsqueeze(0).unsqueeze(0)
        Y = self._forward_transform(Y)

        if sampler_type == "pc":
            sampler = self.get_pc_sampler(predictor, corrector, Y.to(self.device), N=N, 
                corrector_steps=corrector_steps, snr=snr, intermediate=False,
                **kwargs)
        elif sampler_type == "ode":
            sampler = self.get_ode_sampler(Y.to(self.device), N=N, **kwargs)
        else:
            print("{} is not a valid sampler type!".format(sampler_type))
        X_hat, nfe = sampler()
        X_hat[..., : self.data_module.remove_bins, :] = 0. + 1j*0.

        x_hat = self.to_audio(X_hat.squeeze(), T_orig)
        if not do_normalize_online:
            x_hat = x_hat * norm_factor_x
        x_hat = x_hat.squeeze().cpu().numpy()
        
        end = time.time()
        if timeit:
            time_per_frame = (end-start)/num_frames
            return x_hat, time_per_frame
        else:
            return x_hat

    def enhance_STFT_cached(self, Y, sampler_type="pc", predictor="reverse_diffusion_cached",
        corrector="ald", N=30, corrector_steps=1, snr=0.5, output_scale='time', noise_cache={},
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """        
        if sampler_type == "pc":
            sampler = self.get_cached_pc_sampler(predictor, corrector, Y.cuda(), N=N, 
                corrector_steps=corrector_steps, snr=snr, intermediate=False,output_scale=output_scale,
                **kwargs)
        elif sampler_type == "ode":
            sampler = self.get_ode_sampler(Y.cuda(), N=N, **kwargs)
        else:
            print("{} is not a valid sampler type!".format(sampler_type))
        sample, nfe, noise_cache = sampler(noise_cache)
        
        sample = sample.squeeze()
        return sample, noise_cache
    

    def enhance_online(self, y, sampler_type="pc", predictor="reverse_diffusion",
        corrector="ald", N=30, corrector_steps=1, snr=0.5, timeit=False, 
        do_normalize_online=False, vad_threshold=1e-3, noise_cache={},
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        start = time.time()
        T_orig = y.size(1)
        if not do_normalize_online:
            if self.data_module.normalize == "noisy":
                norm_factor_y, norm_factor_x = y.abs().max(), y.abs().max()
            elif self.data_module.normalize in ["noisy_only", "separate"]:
                norm_factor_y, norm_factor_x = y.abs().max(), 1.
            elif self.data_module.normalize == "not":
                norm_factor_y, norm_factor_x = 1., 1.
            y = y / norm_factor_y

        Y = torch.unsqueeze(self._stft(y.to(self.device)), 0)
        Y[..., : self.data_module.remove_bins, :] = 0. + 1j*0.
        Y = pad_spec(Y)
        num_frames = Y.shape[3]

        if do_normalize_online:
            Y = normalize_online(Y.squeeze(), fs=self.data_module.fs, stft_kwargs=self.data_module.stft_kwargs, vad_threshold=vad_threshold)
            Y = Y.unsqueeze(0).unsqueeze(0)
        Y = self._forward_transform(Y)

        if sampler_type == "pc":
            sampler = self.get_cached_pc_sampler(predictor, corrector, Y.to(self.device), N=N, 
                corrector_steps=corrector_steps, snr=snr, intermediate=False,
                **kwargs)
        elif sampler_type == "ode":
            sampler = self.get_ode_sampler(Y.to(self.device), N=N, **kwargs)
        else:
            print("{} is not a valid sampler type!".format(sampler_type))
        X_hat, nfe, noise_cache = sampler(noise_cache)
        X_hat[..., : self.data_module.remove_bins, :] = 0. + 1j*0.

        x_hat = self.to_audio(X_hat.squeeze(), T_orig)
        if not do_normalize_online:
            x_hat = x_hat * norm_factor_x
        x_hat = x_hat.squeeze().cpu().numpy()
        
        end = time.time()
        if timeit:
            time_per_frame = (end-start)/num_frames
            return x_hat, time_per_frame, noise_cache
        else:
            return x_hat, X_hat, noise_cache

