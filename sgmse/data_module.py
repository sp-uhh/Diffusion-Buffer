import os, os.path
from os.path import join
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DataLoader
from glob import glob
from torchaudio import load, info
import numpy as np
import torch.nn.functional as F
import audiomentations as am
from torchaudio.transforms import Resample
import logging

from .util.custom_augmentations import DropSamples, Codec


def get_window(window_type, window_length):
    if window_type == 'sqrthann':
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    elif window_type == 'hann':
        return torch.hann_window(window_length, periodic=True)
    else:
        raise NotImplementedError(
            f"Window type {window_type} not implemented!")


class Specs(Dataset):
    def __init__(self, data_dir, subset, dummy, shuffle_spec, num_frames, 
                 corruption_set="noisy", format="default", normalize="not", spec_transform=None,
                 remove_bins=2, stft_kwargs=None, gains=None, use_clean_norm=False, use_clean_dry=False, **ignored_kwargs):
        subdir = join(data_dir, subset)
        
        # Read file paths according to file naming format.
        if format == "default":
            self.clean_files = sorted(glob(join(data_dir, subset, "clean", "*.wav")))
            self.noisy_files = sorted(glob(join(data_dir, subset, "noisy", "*.wav")))
        elif format == "ears_wham":
            self.clean_files = sorted(glob(join(data_dir, subset, "clean", "**", "*.wav"), recursive=True))
            self.noisy_files = sorted(glob(join(data_dir, subset, "noisy", "**", "*.wav"), recursive=True))
        elif format == "reverb":
            self.clean_files = sorted(glob(join(data_dir, subset, "clean", "**", "*.wav"), recursive=True))
            self.noisy_files = sorted(glob(join(data_dir, subset, "reverberant", "**", "*.wav"), recursive=True))
        else:
            # Feel free to add your own directory format
            raise NotImplementedError(f"Directory format {format} unknown!")
        
        self.num_utterances = len(self.clean_files)
        self.dummy = dummy
        self.num_frames = num_frames
        self.shuffle_spec = shuffle_spec
        self.normalize = normalize
        self.spec_transform = spec_transform if spec_transform is not None else (lambda x: x)
        self.remove_bins = remove_bins
        self.gains = gains
        self.use_clean_norm = use_clean_norm
        self.use_clean_dry = use_clean_dry
        if self.gains is not None and not self.gains == [0, 0]:
            self.gain_augmentation = am.Gain(min_gain_in_db=self.gains[0], max_gain_in_db=self.gains[1], p=1.0)

        assert stft_kwargs is not None and all(k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]), "misconfigured STFT kwargs"
        self.stft_kwargs = stft_kwargs
        self.hop_length = self.stft_kwargs["hop_length"]
        assert self.stft_kwargs.get(
            "center", None) == True, "'center' must be True for current implementation"
        
        self.downsample = None
        self.target_len = (self.num_frames - 1) * self.hop_length

    def __getitem__(self, i):
        inner_index = i % self.num_utterances

        
        x, _ = load(self.clean_files[inner_index])
        y, _ = load(self.noisy_files[inner_index])
        min_len = min(x.size(-1), y.size(-1))
        x, y = x[..., : min_len], y[..., : min_len]




        current_len = x.size(-1)
        #pad to right with target_len many zeros, as this is the buffer length
        x = F.pad(x, (self.target_len, 0), mode='constant')
        y = F.pad(y, (self.target_len, 0), mode='constant')
        current_len = x.size(-1)
        pad = max(self.target_len - current_len, 0)
        
        if pad == 0:
            # extract random part of the audio file
            if self.shuffle_spec:
                start = int(np.random.uniform(0, current_len-self.target_len))
            else:
                start = self.target_len
            x = x[..., start:start+self.target_len]
            y = y[..., start:start+self.target_len]
        else:
            raise ValueError('pad = 0 cannot happen')

        # normalize w.r.t to the noisy or the clean signal or not at all
        # to ensure same clean signal power in x and y.

        if self.normalize == "noisy":
            normfac_y, normfac_x = y.abs().max(), y.abs().max()
        elif self.normalize == "clean":
            normfac_y, normfac_x = x.abs().max(), x.abs().max()
        elif self.normalize == "separate":
            normfac_y, normfac_x = y.abs().max(), x.abs().max()
        elif self.normalize == "noisy_only":
            normfac_y, normfac_x = y.abs().max(), 1.
        elif self.normalize == "not":
            normfac_y, normfac_x = 1., 1.

        x = x / (normfac_x + 1e-8)
        y = y / (normfac_y + 1e-8)

        # Apply random gain (if `gains` was passed at init)
        if self.gains is not None and not self.gains == [0, 0]:
            y = self.gain_augmentation(y, sample_rate=None)
            self.gain_augmentation.freeze_parameters()
            if (not self.use_clean_norm) or (not self.use_clean_dry):
                x = self.gain_augmentation(x, sample_rate=None)  # sample_rate must be passed, but is unused
            self.gain_augmentation.unfreeze_parameters()
            

        X = torch.stft(x, **self.stft_kwargs)
        Y = torch.stft(y, **self.stft_kwargs)
        Y[..., : self.remove_bins, :] = 0 + 0j
        X[..., : self.remove_bins, :] = 0 + 0j

        X, Y = self.spec_transform(X), self.spec_transform(Y)
        return X, Y

    def __len__(self):
        if self.dummy:
            # for debugging shrink the data set size
            # return int(len(self.clean_files)/200)
            return 8
        else:
            return len(self.noisy_files)
        
    def get_eval_data(self, N_files):
        indices = torch.linspace(0, len(self.clean_files)-1, N_files, dtype=torch.int)
        chosen_clean_files = list(self.clean_files[i] for i in indices)
        chosen_noisy_files = list(self.noisy_files[i] for i in indices)
        
        signal_pairs = []
        
        for clean_file, noisy_file in zip(chosen_clean_files, chosen_noisy_files):
            x, _ = load(clean_file)
            y, _ = load(noisy_file)
            min_len = min(x.size(-1), y.size(-1))
            x, y = x[..., : min_len], y[..., : min_len]

            if self.downsample is not None:
                x = self.downsampler(x)
                y = self.downsampler(y)
            
            signal_pairs.append((x,y))
            
        return signal_pairs


class SpecsDataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--base_dir", type=str, required=True,
                            help="The base directory of the dataset. Should contain `train`, `valid` and `test` subdirectories, each of which contain `clean` and `noisy` subdirectories.")
        parser.add_argument("--format", type=str,
                            default="default", help="Read file paths according to file naming format.")
        parser.add_argument("--batch_size", type=int, default=8,
                            help="The batch size. 8 by default.")
        parser.add_argument("--n_fft", type=int, default=638,
                            help="Number of FFT bins. 638 by default.")   # to assure 256 freq bins
        parser.add_argument("--hop_length", type=int, default=160,
                            help="Window hop length. 160 by default.")
        parser.add_argument("--num_frames", type=int, default=256,
                            help="Number of frames for the dataset. 256 by default.")
        parser.add_argument("--window", type=str, choices=("sqrthann", "hann"), default="hann",
                            help="The window function to use for the STFT. 'hann' by default.")
        parser.add_argument("--num_workers", type=int, default=4,
                            help="Number of workers to use for DataLoaders. 4 by default.")
        parser.add_argument("--dummy", action="store_true",
                            help="Use reduced dummy dataset for prototyping.")
        parser.add_argument("--spec_factor", type=float, default=0.15,
                            help="Factor to multiply complex STFT coefficients by. 0.15 by default.")
        parser.add_argument("--spec_abs_exponent", type=float, default=0.5,
                            help="Exponent e for the transformation abs(z)**e * exp(1j*angle(z)). 0.5 by default.")
        parser.add_argument("--normalize", type=str, choices=("clean", "noisy", "noisy_only", "not", "separate"), default="not",
                            help="Normalize the input waveforms by the clean signal, the noisy signal, or not at all. 'not' by default.")
        parser.add_argument("--transform_type", type=str, choices=("exponent", "log", "none"),
                            default="exponent", help="Spectogram transformation for input representation.")
        parser.add_argument("--degrade", action="store_true",
                            help="Augment/degrade data on the fly")
        parser.add_argument("--corruption_set", type=str, default="background_noise",
                            help="Which corruption set (subfolder) to use. 'noisy' by default. Use 'all' to aggregate all non-clean subfolders into one larger set.")
        parser.add_argument("--downsample", type=int, default=None, 
                            help="Downsample inputs to given sampling rate")
        parser.add_argument("--remove_bins", type=int, default=0, 
                            help="FFT bins to remove (DC + 50Hz).")
        parser.add_argument("--fs", type=int, default=16000, 
                            help="Define the sample rate.")
        

        parser.add_argument("--gains", type=int, nargs=2, required=False, default=[0,0],
                            help="If passed, samples a gain value g to apply to both x0 and y (from a uniform distribution with the given bounds in dB, for each data point and call). NOTE: not really a correct model for y=A(g*x0) when the corruption A is nonlinear. We'll use it anyways for training speed")
        parser.add_argument("--use_clean_norm", action="store_true", help="Use normalized clean speech in clean_norm directory (-25dB RMS) as training target.")
        parser.add_argument("--use_clean_dry", action="store_true", help="Use simulated dry speech")


        return parser

    def __init__(
        self, base_dir, format='default', batch_size=8, 
        n_fft=638, hop_length=160, num_frames=256, window='hann',
        num_workers=4, dummy=False, spec_factor=0.15, spec_abs_exponent=0.5,
        gpu=True, normalize='not', transform_type="exponent", degrade=False, 
        downsample=None, fs=32000, remove_bins=2, corruption_set="background_noise", gains=[-12,6], 
        use_clean_norm=False, use_clean_dry=False,
        **kwargs
    ):
        super().__init__()
        self.base_dir = base_dir
        self.format = format
        self.batch_size = batch_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.window = get_window(window, self.n_fft)
        self.windows = {}
        self.num_workers = num_workers
        self.dummy = dummy
        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
        self.gpu = gpu
        self.normalize = normalize
        self.transform_type = transform_type
        self.degrade = degrade
        self.downsample = downsample
        self.kwargs = kwargs
        self.remove_bins = remove_bins
        self.fs = fs
        self.corruption_set = corruption_set
        self.gains = gains
        self.use_clean_norm = use_clean_norm
        self.use_clean_dry = use_clean_dry

    def setup(self, stage=None):
        specs_kwargs = dict(
            data_dir=self.base_dir, dummy=self.dummy, num_frames=self.num_frames,
            stft_kwargs=self.stft_kwargs, spec_transform=self.spec_fwd, downsample=self.downsample,
            format=self.format, normalize=self.normalize, remove_bins=self.remove_bins,
            corruption_set=self.corruption_set, gains=self.gains, use_clean_norm=self.use_clean_norm,
            use_clean_dry=self.use_clean_dry,
            **self.kwargs
        )


        if stage == 'fit' or stage is None:
            self.train_set = Specs(subset='train', shuffle_spec=True, **specs_kwargs)
            self.valid_set = Specs(subset='valid', shuffle_spec=False, **specs_kwargs)
            # assert self.fs == self.valid_set.fs == self.train_set.fs
        if stage == 'test' or stage is None:
            self.test_set = Specs(subset='test', shuffle_spec=False, **specs_kwargs)
                                

    def spec_fwd(self, spec):
        if self.transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                # only do this calculation if spec_exponent != 1, otherwise it's quite a bit of wasted computation
                # and introduced numerical error
                e = self.spec_abs_exponent
                spec = spec.abs()**e * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "none":
            spec = spec
        return spec

    def spec_back(self, spec):
        if self.transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())
        elif self.transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.transform_type == "none":
            spec = spec
        return spec

    @property
    def stft_kwargs(self):
        return {**self.istft_kwargs, "return_complex": True}

    @property
    def istft_kwargs(self):
        return dict(
            n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, center=True
        )

    def _get_window(self, x):
        """
        Retrieve an appropriate window for the given tensor x, matching the device.
        Caches the retrieved windows so that only one window tensor will be allocated per device.
        """
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window

    def stft(self, sig):
        window = self._get_window(sig)
        return torch.stft(sig, **{**self.stft_kwargs, "window": window})

    def istft(self, spec, length=None):
        window = self._get_window(spec)
        return torch.istft(spec, **{**self.istft_kwargs, "window": window, "length": length})

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False
        )
