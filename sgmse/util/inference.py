import torch
from torchaudio import load
from pesq import pesq
from pystoi import stoi
import numpy as np
import librosa
import torchaudio
import torch.nn.functional as F
from .other import si_sdr




def evaluate_model(model, num_eval_files, diff_buffer_length, num_frames, spec=True, audio=True, fixed_buffer_steps='lin'):
    signal_pairs = model.data_module.valid_set.get_eval_data(num_eval_files)
    fs = model.data_module.fs

    basic_metrics = {
        "si_sdr": np.zeros(num_eval_files),
        "estoi": np.zeros(num_eval_files),
        "pesq": np.zeros(num_eval_files)
    }


    
    other_metrics = None

    spec_list = {"y": [], "x_hat": [], "x": []} if spec else None      
    audio_list = {"y": [], "x_hat": [], "x": []} if audio else None      
    
    if model.loss_type in ['gencon_v1', 'gencon_v2', 'gencon_mse', 'gencon_mrstft', 'gencon_mel']:
        basic_metrics['pesq_cond'] = np.zeros(num_eval_files)
        audio_list['cond'] = []
        spec_list['cond'] = []

    for i, (x, y) in enumerate(signal_pairs):
        # Determine inference path based on loss type
        # Load wavs
        #x, fs = load(clean_file)
        #y, fs = load(noisy_file)
        #fn = noisy_file.split('/')[-1]

        
        T_orig = x.size(1)   
        #sr = fs

        # Normalize per utterance
        normfac = 1.0
            
        y = y / normfac
        x = x / normfac

        # Prepare DNN input
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        enhanced = torch.zeros_like(Y, device=Y.device)
        enhanced_cond = torch.zeros_like(Y, device=Y.device)

        
        X = torch.unsqueeze(model._forward_transform(model._stft(x.cuda())), 0)
        
        #pad Y from the left with gate_length many 0s
        Y = F.pad(Y, (num_frames-1, diff_buffer_length), "constant", 0)
        X = F.pad(X, (num_frames-1, diff_buffer_length), "constant", 0)
        
        EnvironNoise = Y - X

        
        #linspace for diffimes from 0.03 to 1 with diff_gate_length steps
        if fixed_buffer_steps == 'lin':
            diff_times = torch.linspace(model.t_eps, model.sde.T, diff_buffer_length, device=Y.device)
        elif fixed_buffer_steps == 'power':
            diff_times = torch.linspace(model.t_eps, model.sde.T, diff_buffer_length, device=Y.device)
            diff_times = diff_times**0.1
            diff_times[0] = model.t_eps

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
        ot_gt_target = torch.zeros_like(Y[..., :num_frames], device=Y.device)
        correct_mean_evo = torch.zeros_like(Y[..., :num_frames], device=Y.device)
        for i_frame in range(num_frames, Y.size(-1) - 1):
            current_chunk = Y[..., i_frame - num_frames:i_frame]
            current_GT = X[..., i_frame - num_frames:i_frame]
            
            mean_evo_GT = (1-mean_evo_interpolation)*current_GT + mean_evo_interpolation*current_GT

            
            current_env = EnvironNoise[..., i_frame - num_frames:i_frame]
            mask_init[..., -init_iteration_var] = 1
            
            perturbed_input[..., :-1] = perturbed_input[..., 1:].clone()
            ot_gt_target[..., :-1] = ot_gt_target[..., 1:].clone()
            correct_mean_evo[..., :-1] = correct_mean_evo[..., 1:].clone()
            #std_score[..., :-1] = std_score[..., 1:].clone()
            
            if i_frame==num_frames:
                z = torch.randn_like(current_chunk, device=Y.device)
                if model.bwr_index > 0: 
                    z[:,:, :model.bwr_index,:] = 0
                noise = z*std*mask
                perturbed_input = perturbed_input +  noise
                ot_gt_target = current_env + noise
                #correct_mean_evo = mean_evo_GT + noise
                #score_gt = z/std_score**2

            z = torch.randn_like(current_chunk[..., -1], device=Y.device)
            noise = z * std_at_rsp
            
            if model.bwr_index > 0: 
                noise[:,:, :model.bwr_index] = 0
                
            perturbed_input[..., -1] = current_chunk[..., -1] + noise
            ot_gt_target[..., -1] = current_env[..., -1] + noise


            

            
            
            if init_iteration_var <= diff_buffer_length and i_frame != num_frames:
                perturbed_input = perturbed_input*mask_init
                mask_diff = mask - mask_init #empty part of the gate
                z = torch.randn_like(current_chunk, device=Y.device)*mask_diff
                if model.bwr_index > 0: 
                    z[:,:, :model.bwr_index,:] = 0
                perturbed_input = perturbed_input + z*std*mask_diff
                correct_mean_evo = correct_mean_evo*mask_init + z*std*mask_diff
                #score_gt[:,:,:, -diff_gate_length:init_iteration_var] = z[:,:,:, -diff_gate_length:init_iteration_var]/std_score[:,:,:, -diff_gate_length:init_iteration_var]**2
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
            

            if model.loss_type in ['ot', 'ot_time']:
                vectfield = model.forward(perturbed_input, diff_times[None, :], current_chunk, scale)
                #use GT information
                #vectfield = ot_gt_target
                gate = perturbed_input.clone() - vectfield*stepsizes*mask
                perturbed_input[..., -diff_buffer_length:] = gate[..., -diff_buffer_length:]
                #audio_tmp = model.to_audio(perturbed_input.squeeze(), model.target_len)
                #torchaudio.save(f'{i_frame}_per.wav', audio_tmp.unsqueeze(0).cpu(), sr)
            elif model.loss_type in ['mse', 'gencon_mse', 'mse_pred']:
                f, g = model.sde.sde(perturbed_input, diff_times[None, None, None, :], current_chunk)
                if model.loss_type == 'gencon_mse':
                    dnn_in  = [perturbed_input, diff_times[None, :], current_chunk, scale]
                    score, Con_estimate = model.dnn(*dnn_in)
                    score = score*mask
                    #score, _ = model.forward(perturbed_input, diff_times[None, :], current_chunk, scale)*mask
                else:
                    score = model.forward(perturbed_input, diff_times[None, :], current_chunk, scale)*mask

                    
                perturbed_input_gate_part = perturbed_input.clone()
                perturbed_input[..., -diff_buffer_length:] = perturbed_input_gate_part[..., -diff_buffer_length:] - (f[..., -diff_buffer_length:] - g[..., -diff_buffer_length:]**2*score[..., -diff_buffer_length:])*stepsizes[..., -diff_buffer_length:]
                z = torch.randn_like(perturbed_input)*mask
                z[..., -diff_buffer_length] = 0
                if model.bwr_index > 0: 
                    z[:,:, :model.bwr_index, :] = 0
                
                perturbed_input = perturbed_input + z*g*torch.sqrt(stepsizes)
                
                
                
            elif model.loss_type in ['predictive', 'onestep']:
                GT_estimate = model.forward(perturbed_input, diff_times[None, :], current_chunk, scale)
                #use GT information
                #GT_estimate = X[..., i_frame - num_frames:i_frame]
                perturbed_input = (GT_estimate + (current_chunk - GT_estimate)*mean_evo_shifted_interpolation[None,None,None,:])
                z= torch.randn_like(current_chunk, device=Y.device)
                perturbed_input = perturbed_input + std_shifted*z
            elif model.loss_type in ['gencon_v1', 'gencon_v2', 'gencon_mrstft', 'gencon_mel']:
                GT_estimate, Con_estimate = model.forward(perturbed_input, diff_times[None, :], current_chunk, scale)
                #use GT information
                #GT_estimate = X[..., i_frame - num_frames:i_frame]
                perturbed_input = (GT_estimate + (current_chunk - GT_estimate)*mean_evo_shifted_interpolation[None,None,None,:])
                z= torch.randn_like(current_chunk, device=Y.device)
                perturbed_input = perturbed_input + std_shifted*z
            elif model.loss_type == 'mean_evo':
                #means = mean_evo_GT #GT data
                means = model.forward(perturbed_input, diff_times[None, :], current_chunk, scale)*mask
                environ_noise= mask*(current_chunk - means)/(1-mean_evo_interpolation[None,None,None,:])
                means_shifted = current_chunk  - (1 - mean_evo_shifted_interpolation[None,None,None,:])*environ_noise
                gate = means_shifted*mask + std_shifted*torch.randn_like(means, device=Y.device)*mask
                perturbed_input[..., -diff_buffer_length:] = gate[..., -diff_buffer_length:].clone()
                
                #audio_tmp = model.to_audio(perturbed_input.squeeze(), model.target_len)
                #torchaudio.save(f'{i_frame}_per.wav', audio_tmp.unsqueeze(0).cpu(), sr)
            else:
                raise ValueError('loss type not implemented!')
            
            if init_iteration_var < diff_buffer_length:
                init_iteration_var += 1
            else:
                if model.loss_type == 'predictive':
                    #works only with predictive loss atm
                    out_frame = GT_estimate[..., -diff_buffer_length]
                else:
                    out_frame = perturbed_input[..., -diff_buffer_length]
                    #out_frame[:,:, :model.bwr_index] = current_chunk[:,:, :model.bwr_index, -diff_buffer_length]
                enhanced[..., i_frame - num_frames - diff_buffer_length + 1] = out_frame
                if model.loss_type in ['gencon_v1', 'gencon_v2', 'gencon_mse', 'gencon_mrstft', 'gencon_mel']:
                    enhanced_cond[..., i_frame - num_frames - diff_buffer_length + 1] = Con_estimate[..., -diff_buffer_length]
        # Convert references to numpy
        x_np = x.squeeze().cpu().numpy()
        y_np = y.squeeze().cpu().numpy()
        x_hat = model.to_audio(enhanced.squeeze(), T_orig)
        x_hat = x_hat * normfac
        x_hat = x_hat.cpu().numpy()

        if model.loss_type in ['gencon_v1', 'gencon_v2', 'gencon_mse', 'gencon_mrstft', 'gencon_mel']:
            x_hat_cond = model.to_audio(enhanced_cond.squeeze(), T_orig)
            x_hat_cond = x_hat_cond * normfac
            x_hat_cond = x_hat_cond.cpu().numpy()
            x_16k_cond = librosa.resample(x_hat_cond, orig_sr=fs, target_sr=16000) if fs != 16000 else x_hat_cond
        # Resample to 16 kHz for PESQ and DNSMOS
        x_16k = librosa.resample(x_np, orig_sr=fs, target_sr=16000) if fs != 16000 else x_np
        x_hat_16k = librosa.resample(x_hat, orig_sr=fs, target_sr=16000) if fs != 16000 else x_hat

        # Compute basic metrics
        basic_metrics["si_sdr"][i] = si_sdr(x_np, x_hat)
        try:
            basic_metrics["pesq"][i] = pesq(16000, x_16k, x_hat_16k, 'wb')
        except (ValueError, TypeError):  # catches both NaN and other potential type errors
            basic_metrics["pesq"][i] = 0
        basic_metrics["estoi"][i] = stoi(x_np, x_hat, fs, extended=True)
        
        if model.loss_type in ['gencon_v1', 'gencon_v2', 'gencon_mse', 'gencon_mrstft', 'gencon_mel']:
            try:
                basic_metrics["pesq_cond"][i] = pesq(16000, x_16k, x_16k_cond, 'wb')
            except (ValueError, TypeError):  # catches both NaN and other potential type errors
                basic_metrics["pesq_cond"][i] = 0

        # Collect specimens for visualization
        if i <  num_eval_files//2:
            if spec:
                spec_list["y"].append(model._stft(torch.from_numpy(y_np)))
                spec_list["x_hat"].append(model._stft(torch.from_numpy(x_hat)))
                spec_list["x"].append(model._stft(torch.from_numpy(x_np)))
                if model.loss_type in ['gencon_v1', 'gencon_v2', 'gencon_mse', 'gencon_mrstft', 'gencon_mel']:
                    spec_list["cond"].append(model._stft(torch.from_numpy(x_hat_cond)))
            if audio:
                audio_list["y"].append(y_np)
                audio_list["x_hat"].append(x_hat)
                audio_list["x"].append(x_np)
                if model.loss_type in ['gencon_v1', 'gencon_v2', 'gencon_mse', 'gencon_mrstft', 'gencon_mel']:
                    audio_list["cond"].append(x_hat_cond)

    return basic_metrics, other_metrics, spec_list, audio_list
