import glob
from soundfile import write
import torchaudio
from tqdm import tqdm
from pesq import pesq
from torchaudio import load
import torch.nn.functional as F
from compute_fadtk import FADTK_embedding_cacher
import distillmos
#from DNSMOS.dnsmos_local import dnsmos
import torch
from argparse import ArgumentParser
from os.path import join
import pandas as pd
from sgmse.model import ScoreModel
from wvmos import get_wvmos
from pesq import pesq
from pystoi import stoi
from utils import log_spectral_distance
from utils import energy_ratios, ensure_dir, print_mean_std



# backup the original torch.load
original_torch_load = torch.load

# patch torch.load globally
def patched_torch_load(*args, **kwargs):
    # force weights_only=False for all loads
    kwargs.setdefault('weights_only', False)
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load







def log_spectra(x, x_hat):
    X_stft = model._stft(torch.tensor(x).cuda())
    Xhat_stft = model._stft(torch.tensor(x_hat).cuda())
    diff = (torch.log(torch.abs(X_stft)) - torch.log(torch.abs(Xhat_stft)))**2
    L2 = torch.sqrt(torch.sum(diff))/diff.shape[1]
    log_spectra = L2.detach().cpu().numpy()
    return float(log_spectra)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--destination_folder", type=str, help="baseline or linear.")
    parser.add_argument("--experiments_folder", type=str, required=True, help='Directory to save the results.')
    parser.add_argument("--test_dir", type=str, required=True, help='Directory containing the test data')
    parser.add_argument("--reverse_starting_point", type=float, default=1.0, required=True, help='Directory containing the test data')
    parser.add_argument("--ckpt", type=str, help='Path to model checkpoint.')
    parser.add_argument("--N", type=int, default=30, help="Number of reverse steps")
    parser.add_argument("--num_frames", type=int, default=128, help="Number of frames to the NN")
    parser.add_argument("--timestep_type", type=str, default='linear', help="timestep for sampling")
    args = parser.parse_args()

    clean_dir = join(args.test_dir, "test", "clean")
    noisy_dir = join(args.test_dir, "test", "noisy")

    checkpoint_file = args.ckpt

    target_dir = args.experiments_folder + "/{}/".format(
        args.destination_folder)


    ensure_dir(target_dir + "files/")

    # Settings
    sr = 16000
    N = args.N #also the buffer length
    diff_buffer_length = N
    timestep_type = args.timestep_type
    num_frames = args.num_frames
    rsp = args.reverse_starting_point
    sqa_model = distillmos.ConvTransformerSQAModel()
    sqa_model = sqa_model.to('cuda')

    wvmos_model = get_wvmos(cuda=True)

    # Load score model
    model = ScoreModel.load_from_checkpoint(
        checkpoint_file, base_dir="",
        batch_size=16, num_workers=0, kwargs=dict(gpu=False)
    )
    model.eval(no_ema=False)
    model.cuda()
    
    noisy_files = sorted(glob.glob('{}/**/*.wav'.format(noisy_dir)))
    freq_weight = 1
    target_len =  (num_frames - 1) * model.data_module.hop_length
    with torch.no_grad():
        data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": [], "si_sir": [], "si_sar": [],
                "WVMOS": [], "LSD": [], "DistillMOS": []}
        fadtk_cacher = FADTK_embedding_cacher([], clean_dir, diff_buffer_length)
        for cnt, noisy_file in tqdm(enumerate(noisy_files)):
            filename =  '/'.join(noisy_file.split('/')[-2:])

            # Load wav
            x, fs = load(join(clean_dir, filename))
            y, fs = load(noisy_file)
            
            
            #requires only for BWE as the dataset has different length of clean and noisy files
            if x.shape[1] != y.shape[1]:
                len_wav = min(x.shape[1], y.shape[1])
                x = x[:, :len_wav]
                y = y[:, :len_wav]
                
            
            if fs != sr:
                x = torchaudio.functional.resample(x, fs, sr)
                y = torchaudio.functional.resample(y, fs, sr)
            
            T_orig = x.size(1)   
            #sr = fs

            # Normalize per utterance
            if model.data_module.normalize == "noisy":
                normfac = y.abs().max()
            elif model.data_module.normalize == "clean":
                normfac = x.abs().max()
            elif model.data_module.normalize == "not":
                normfac = 1.0
                
            y = y / normfac
            x = x / normfac

                   # Prepare DNN input
            Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
            enhanced = torch.zeros_like(Y, device=Y.device)
            X = torch.unsqueeze(model._forward_transform(model._stft(x.cuda())), 0)
            
            #pad Y from the left with diff_buffer_length many 0s
            Y = F.pad(Y, (num_frames-1, diff_buffer_length), "constant", 0)
            X = F.pad(X, (num_frames-1, diff_buffer_length), "constant", 0)
            
            EnvironNoise = Y - X

            
            #linspace for diffimes from t_eps to 1 with diff_buffer_length steps
            diff_times = torch.linspace(model.t_eps, rsp, diff_buffer_length, device=Y.device)
            diff_times = F.pad(diff_times, (num_frames - diff_buffer_length, 0), mode='constant', value=0)
            diff_times_shifted = diff_times.clone()
            diff_times_shifted[1:] = diff_times[:-1]
            diff_times_shifted[0] = 0
            
            mean_evo_interpolation = model.sde.get_meanevo_factor(diff_times)
            mean_evo_shifted_interpolation = model.sde.get_meanevo_factor(diff_times_shifted)

            std = model.sde.get_diff_std(num_frames, diff_times).to(device=Y.device)
            std_shifted = model.sde.get_diff_std(num_frames,  diff_times_shifted).to(device=Y.device)
            std_at_rsp = std[..., -1]
            stepsizes = torch.zeros_like(std)
            stepsizes[..., -(diff_buffer_length-1):] = diff_times[-1] - diff_times[-2]
            stepsizes[..., -diff_buffer_length] = diff_times[-diff_buffer_length]
            mask = torch.zeros_like(std, device=Y.device)
            mask[:,:,:,-diff_buffer_length:] = 1
            init_iteration_var = 1
            output_scale = model.output_scale
            mask_init = torch.zeros_like(std, device=Y.device)
            perturbed_input = torch.zeros_like(Y[..., :num_frames], device=Y.device)

            score_gt = torch.zeros_like(Y[..., :num_frames], device=Y.device)
            std_score = std
            std_score[..., :-diff_buffer_length] = 1 #for divsion by 0
            for i_frame in range(num_frames, Y.size(-1) - 1):
                current_chunk = Y[..., i_frame - num_frames:i_frame]

            
                mask_init[..., -init_iteration_var] = 1
                perturbed_input[..., :-1] = perturbed_input[..., 1:].clone()
                std_score[..., :-1] = std_score[..., 1:].clone()
                
                if i_frame==num_frames:
                    z = torch.randn_like(current_chunk, device=Y.device)
                    noise = z*std*mask
                    perturbed_input = perturbed_input +  noise
                    score_gt = z/std_score**2

                z = torch.randn_like(current_chunk[..., -1], device=Y.device)
                noise = z * std_at_rsp
                perturbed_input[..., -1] = current_chunk[..., -1] + noise

                score_gt[..., -1] = z/std_score[..., -1]**2
                
                if init_iteration_var <= diff_buffer_length and i_frame != num_frames:
                    perturbed_input = perturbed_input*mask_init
                    mask_diff = mask - mask_init #empty part of the buffer
                    z = torch.randn_like(current_chunk, device=Y.device)*mask_diff
                    perturbed_input = perturbed_input + z*std*mask_diff
                    score_gt[:,:,:, -diff_buffer_length:init_iteration_var] = z[:,:,:, -diff_buffer_length:init_iteration_var]/std_score[:,:,:, -diff_buffer_length:init_iteration_var]**2
                    if init_iteration_var == diff_buffer_length:
                        init_iteration_var = init_iteration_var + 1
                #scale shape [BS, 1, Freq, NumFrames]
                if output_scale=='time':
                    scale = diff_times[None, None, None, :]
                    scale[:,:,:,:-diff_buffer_length] = 1
                elif output_scale=='sigma':
                    #not supported atm
                    pass
                    scale = std
                elif output_scale=='no':
                    scale = torch.ones_like(std)
                else:
                    raise ValueError('output scale not implemented!')
                scale = scale.to(device=Y.device)
                #reverse process
                


                if model.loss_type == 'mse': #denoising score matching loss
                    f, g = model.sde.sde(perturbed_input, diff_times[None, None, None, :], current_chunk)
                    score = model.forward(perturbed_input, diff_times[None, :], current_chunk, scale)*mask  #or used score_gt for debugging
                    perturbed_input_buffer_part = perturbed_input.clone()
                    perturbed_input[..., -diff_buffer_length:] = perturbed_input_buffer_part[..., -diff_buffer_length:] - (f[..., -diff_buffer_length:] - g[..., -diff_buffer_length:]**2*score[..., -diff_buffer_length:])*stepsizes[..., -diff_buffer_length:]
                    z = torch.randn_like(perturbed_input)*mask
                    z[..., -diff_buffer_length] = 0
                    perturbed_input = perturbed_input + z*g*torch.sqrt(stepsizes)
                else:
                    raise ValueError('loss type not implemented!')
                
                if init_iteration_var < diff_buffer_length:
                    init_iteration_var += 1
                else:
                    out_frame = perturbed_input[..., -diff_buffer_length]
                    enhanced[..., i_frame - num_frames - diff_buffer_length + 1] = out_frame

            y = y * normfac
            x = x * normfac

            x_hat = model.to_audio(enhanced.squeeze(), T_orig)
            x_hat = x_hat * normfac
            x_hat_tensor = x_hat

            x_hat = x_hat.cpu().numpy()
            x = x.squeeze().cpu().numpy()
            y = y.squeeze().cpu().numpy()
            n = y - x
            

            # Write enhanced wav file
            ensure_dir(target_dir + "files/" + filename.split('/')[-2] + '/')
            write(target_dir + "files/" + filename, x_hat, 16000)


            data["filename"].append(filename)
            with torch.no_grad():
                mos = sqa_model(x_hat_tensor.unsqueeze(0))
            data["DistillMOS"].append(mos.squeeze().detach().cpu().numpy().item())
            try:
                p = pesq(sr, x, x_hat, 'wb')
            except:  
                p = float("nan")
            #data["log_spectra"].append(log_spectra(x, x_hat))
            data["pesq"].append(p)
            data["estoi"].append(stoi(x, x_hat, sr, extended=True))
            data["si_sdr"].append(energy_ratios(x_hat, x, n)[0])
            data["si_sir"].append(energy_ratios(x_hat, x, n)[1])
            data["si_sar"].append(energy_ratios(x_hat, x, n)[2])
            lsd_norm = log_spectral_distance(torch.tensor(x_hat), torch.tensor(x))
            data["LSD"].append(lsd_norm)
            fadtk_cacher.compute_embd_delay(f'{diff_buffer_length}', x_hat_tensor)
    
            wvmos = wvmos_model.calculate_one(target_dir + "files/" + filename)
            data["WVMOS"].append(wvmos)

        # Save results as DataFrame

        df = pd.DataFrame(data)
        df.to_csv(join(target_dir, "_results.csv"), index=False)

        fad_value = fadtk_cacher.compute_fad_value(f'{diff_buffer_length}')

        # Save average results
        text_file = join(target_dir, "_avg_results.txt")
        with open(text_file, 'w') as file:
            file.write("PESQ: {} \n".format(print_mean_std(data["pesq"])))
            file.write("ESTOI: {} \n".format(print_mean_std(data["estoi"])))
            file.write("SI-SDR: {} \n".format(print_mean_std(data["si_sdr"])))
            file.write("SI-SIR: {} \n".format(print_mean_std(data["si_sir"])))
            file.write("SI-SAR: {} \n".format(print_mean_std(data["si_sar"])))
            file.write("WVMOS: {} \n".format(print_mean_std(data["WVMOS"])))
            file.write("LSD: {} \n".format(print_mean_std(data["LSD"])))
            file.write("Distill MOS: {} \n".format(print_mean_std(data["DistillMOS"])))
            file.write("FAD: {} \n".format(fad_value))



        # Save settings
        text_file = join(target_dir, "_settings.txt")
        with open(text_file, 'w') as file:
            file.write("test dir: {}\n".format(args.test_dir))
            file.write("checkpoint file: {}\n".format(checkpoint_file))
            file.write("reverse starting point: {}\n".format(rsp))
            file.write("N: {}\n".format(N))
            file.write("timestep type: {}\n".format(timestep_type))
