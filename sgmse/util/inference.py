import torch
import torch.nn.functional as F
from pesq import pesq
from torchaudio import load
import librosa
from pystoi import stoi
import numpy as np
import torchaudio
import librosa
from .other import si_sdr

# Settings
sr = 16000
snr = 0.5
corrector_steps = 0


def evaluate_model(model, num_eval_files, spec=True, audio=True, num_frames=128, diff_buffer_length=10):
    clean_files = model.data_module.valid_set.clean_files
    noisy_files = model.data_module.valid_set.noisy_files

    fs = model.data_module.fs
    total_num_files = len(clean_files)
    indices = torch.linspace(0, total_num_files-1, num_eval_files, dtype=torch.int)
    clean_files = list(clean_files[i] for i in indices)
    noisy_files = list(noisy_files[i] for i in indices)
    spec_list = {"y": [], "x_hat": [], "x": [], "fn": []} if spec else None      
    audio_list = {"y": [], "x_hat": [], "x": [], "fn": []} if audio else None   
    basic_metrics = {"si_sdr": np.zeros(num_eval_files), "estoi": np.zeros(num_eval_files), "pesq": np.zeros(num_eval_files)}
        

    # iterate over files
    i = 0
    for (clean_file, noisy_file) in zip(clean_files, noisy_files):
        # Load wavs
        x, fs = load(clean_file)
        y, fs = load(noisy_file)
        fn = noisy_file.split('/')[-1]
        
        
        if x.shape != y.shape:
            _min = np.min([x.shape[-1], y.shape[-1]])
            x = x[:, :_min]
            y = y[:, :_min]

        
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
        
        #pad Y from the left with buffer_length many 0s
        Y = F.pad(Y, (num_frames-1, diff_buffer_length), "constant", 0)
        X = F.pad(X, (num_frames-1, diff_buffer_length), "constant", 0)
        
        EnvironNoise = Y - X

        
        #linspace for diffimes from 0.03 to 1 with diff_buffer_length steps
        diff_times = torch.linspace(model.t_eps, model.sde.T, diff_buffer_length, device=Y.device)
        diff_times = F.pad(diff_times, (num_frames - diff_buffer_length, 0), mode='constant', value=0)
        diff_times_shifted = diff_times.clone()
        diff_times_shifted[1:] = diff_times[:-1]
        diff_times_shifted[0] = 0
        
        #mean_evo_interpolation = model.sde.get_meanevo_factor(diff_times)
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
        ot_gt_target = torch.zeros_like(Y[..., :num_frames], device=Y.device)
        correct_mean_evo = torch.zeros_like(Y[..., :num_frames], device=Y.device)
        for i_frame in range(num_frames, Y.size(-1) - 1):
            current_chunk = Y[..., i_frame - num_frames:i_frame]
            
            #For debugging, get GT and GT mean
            #current_GT = X[..., i_frame - num_frames:i_frame]
            #mean_evo_GT = (1-mean_evo_interpolation)*current_GT + mean_evo_interpolation*current_GT

            
            current_env = EnvironNoise[..., i_frame - num_frames:i_frame]
            mask_init[..., -init_iteration_var] = 1
            
            perturbed_input[..., :-1] = perturbed_input[..., 1:].clone()
            ot_gt_target[..., :-1] = ot_gt_target[..., 1:].clone()
            correct_mean_evo[..., :-1] = correct_mean_evo[..., 1:].clone()
            #std_score[..., :-1] = std_score[..., 1:].clone()
            
            if i_frame==num_frames:
                z = torch.randn_like(current_chunk, device=Y.device)
                noise = z*std*mask
                perturbed_input = perturbed_input +  noise
                ot_gt_target = current_env + noise


            z = torch.randn_like(current_chunk[..., -1], device=Y.device)
            noise = z * std_at_rsp
            perturbed_input[..., -1] = current_chunk[..., -1] + noise
            ot_gt_target[..., -1] = current_env[..., -1] + noise

            
            if init_iteration_var <= diff_buffer_length and i_frame != num_frames:
                perturbed_input = perturbed_input*mask_init
                mask_diff = mask - mask_init #empty part of the buffer
                z = torch.randn_like(current_chunk, device=Y.device)*mask_diff
                perturbed_input = perturbed_input + z*std*mask_diff
                correct_mean_evo = correct_mean_evo*mask_init + z*std*mask_diff
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
            

            if model.loss_type == 'mse':
                f, g = model.sde.sde(perturbed_input, diff_times[None, None, None, :], current_chunk)
                score = model.forward(perturbed_input, diff_times[None, :], current_chunk, scale)*mask
                
                perturbed_input_buffer_part = perturbed_input.clone()
                perturbed_input[..., -diff_buffer_length:] = perturbed_input_buffer_part[..., -diff_buffer_length:] - (f[..., -diff_buffer_length:] - g[..., -diff_buffer_length:]**2*score[..., -diff_buffer_length:])*stepsizes[..., -diff_buffer_length:]
                z = torch.randn_like(perturbed_input)*mask
                z[..., -diff_buffer_length] = 0
                perturbed_input = perturbed_input + z*g*torch.sqrt(stepsizes)
            elif model.loss_type in ['data_pred_loss']:
                GT_estimate = model.forward(perturbed_input, diff_times[None, :], current_chunk, scale)
                #use GT information for debugging
                #GT_estimate = X[..., i_frame - num_frames:i_frame]
                perturbed_input = (GT_estimate + (current_chunk - GT_estimate)*mean_evo_shifted_interpolation[None,None,None,:])
                z= torch.randn_like(current_chunk, device=Y.device)
                perturbed_input = perturbed_input + std_shifted*z
            else:
                raise ValueError('loss type not implemented!')
            
            if init_iteration_var < diff_buffer_length:
                init_iteration_var += 1
            else:
                if model.loss_type == 'data_pred_loss':
                    #is the same as taking 
                    # out_frame = perturbed_input[..., -diff_buffer_length]
                    out_frame = GT_estimate[..., -diff_buffer_length]
                else:
                    out_frame = perturbed_input[..., -diff_buffer_length]
                enhanced[..., i_frame - num_frames - diff_buffer_length + 1] = out_frame



        y = y * normfac
        x = x * normfac
        x_hat = model.to_audio(enhanced.squeeze(), T_orig)
        x_hat = x_hat * normfac

        
        x_hat = x_hat.cpu().numpy()
        x = x.squeeze().cpu().numpy()
        y = y.squeeze().cpu().numpy()
        
        # Both PESQ and DNSMOS need 16khz so we resample here once if necesary
        x_16k = librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x
        x_hat_16k = librosa.resample(x_hat, orig_sr=fs, target_sr=16000) if sr != 16000 else x_hat

        basic_metrics["si_sdr"][i] = si_sdr(x, x_hat)
        try:
            p = pesq(sr, x_16k, x_hat_16k, 'wb')
        except:
            p = 1.0
        basic_metrics["pesq"][i] = p
        basic_metrics["estoi"][i] = stoi(x, x_hat, fs, extended=True)


        if i < num_eval_files//2:
            if spec:
                spec_list["y"].append(model._stft(torch.from_numpy(y)))
                spec_list["x_hat"].append(model._stft(torch.from_numpy(x_hat)))
                spec_list["x"].append(model._stft(torch.from_numpy(x)))
                spec_list["fn"].append(fn)
            if audio:
                audio_list["y"].append(y)
                audio_list["x_hat"].append(x_hat)
                audio_list["x"].append(x)
                audio_list["fn"].append(fn)
        i = i+1

        
    return basic_metrics, spec_list, audio_list


