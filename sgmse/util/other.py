import os
import numpy as np
import scipy.stats
from scipy.signal import butter, sosfilt

import torch

from pesq import pesq
from pystoi import stoi
import torchaudio 

### DNSMOS imports
import numpy.polynomial.polynomial as poly
import pandas as pd
from requests import session
from tqdm import tqdm
import librosa

### NISQA imports
#from .NISQA.nisqa.NISQA_model import nisqaModel
#from .NISQA.nisqa.NISQA_lib import SpeechQualityDataset

def si_sdr_components(s_hat, s, n):
    """
    """
    # s_target
    alpha_s = np.dot(s_hat, s) / np.linalg.norm(s)**2
    s_target = alpha_s * s

    # e_noise
    alpha_n = np.dot(s_hat, n) / np.linalg.norm(n)**2
    e_noise = alpha_n * n

    # e_art
    e_art = s_hat - s_target - e_noise
    
    return s_target, e_noise, e_art

def energy_ratios(s_hat, s, n):
    """
    """
    s_target, e_noise, e_art = si_sdr_components(s_hat, s, n)

    si_sdr = 10*np.log10(np.linalg.norm(s_target)**2 / np.linalg.norm(e_noise + e_art)**2)
    si_sir = 10*np.log10(np.linalg.norm(s_target)**2 / np.linalg.norm(e_noise)**2)
    si_sar = 10*np.log10(np.linalg.norm(s_target)**2 / np.linalg.norm(e_art)**2)

    return si_sdr, si_sir, si_sar

def mean_conf_int(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

class Method():
    def __init__(self, name, base_dir, metrics):
        self.name = name
        self.base_dir = base_dir
        self.metrics = {} 
        
        for i in range(len(metrics)):
            metric = metrics[i]
            value = []
            self.metrics[metric] = value 
            
    def append(self, matric, value):
        self.metrics[matric].append(value)

    def get_mean_ci(self, metric):
        return mean_conf_int(np.array(self.metrics[metric]))

def hp_filter(signal, cut_off=80, order=10, sr=16000):
    factor = cut_off /sr * 2
    sos = butter(order, factor, 'hp', output='sos')
    filtered = sosfilt(sos, signal)
    return filtered

def si_sdr(s, s_hat):
    alpha = np.dot(s_hat, s)/np.linalg.norm(s)**2   
    sdr = 10*np.log10(np.linalg.norm(alpha*s)**2/np.linalg.norm(
        alpha*s - s_hat)**2)
    return sdr

def snr_dB(s,n):
    s_power = 1/len(s)*np.sum(s**2)
    n_power = 1/len(n)*np.sum(n**2)
    snr_dB = 10*np.log10(s_power/n_power)
    return snr_dB

def pad_spec(Y):
    T = Y.size(3)
    if T%64 !=0:
        num_pad = 64-T%64
    else:
        num_pad = 0
    pad2d = torch.nn.ZeroPad2d((0, num_pad, 0,0))
    return pad2d(Y)


def ensure_dir(file_path):
    directory = file_path
    if not os.path.exists(directory):
        os.makedirs(directory)


def print_metrics(x, y, x_hat_list, labels, sr=16000):
    _si_sdr_mix = si_sdr(x, y)
    _pesq_mix = pesq(sr, x, y, 'wb')
    _estoi_mix = stoi(x, y, sr, extended=True)
    print(f'Mixture:  PESQ: {_pesq_mix:.2f}, ESTOI: {_estoi_mix:.2f}, SI-SDR: {_si_sdr_mix:.2f}')
    for i, x_hat in enumerate(x_hat_list):
        _si_sdr = si_sdr(x, x_hat)
        _pesq = pesq(sr, x, x_hat, 'wb')
        _estoi = stoi(x, x_hat, sr, extended=True)
        print(f'{labels[i]}: {_pesq:.2f}, ESTOI: {_estoi:.2f}, SI-SDR: {_si_sdr:.2f}')

def mean_std(data):
    data = data[~np.isnan(data)]
    mean = np.mean(data)
    std = np.std(data)
    return mean, std

def print_mean_std(data, decimal=2):
    data = np.array(data)
    data = data[~np.isnan(data)]
    mean = np.mean(data)
    std = np.std(data)
    if decimal == 2:
        string = f'{mean:.2f} ± {std:.2f}'
    elif decimal == 1:
        string = f'{mean:.1f} ± {std:.1f}'
    return string


class NISQAScore:
    def __init__(self):

        args = {"mode":"predict_file", "deg": "/data/lemercier/databases/wsj0+chime_julian/audio/train/noisy/20ma0113_1329_12.7.wav", "pretrained_model":"sgmse/util/NISQA/weights/nisqa_mos_only.tar", "bs":1, "ms_channel":1}
        self.model = nisqaModel(args)

    def compute_one(self, audio, fs):
        SAMPLING_RATE = 48000
        self.model.args["deg"] = audio
        
        if fs != SAMPLING_RATE:
            audio = librosa.resample(audio, orig_sr=fs, target_sr=SAMPLING_RATE)

        x_spec = torch.from_numpy(NISQAScore.get_librosa_melspec(audio, sr=fs)[np.newaxis, :, np.newaxis, :, np.newaxis])
        n_wins = torch.Tensor([x_spec.shape[-1]]).cpu()
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(
                x_spec,
                n_wins,
                batch_first=True,
                enforce_sorted=False
                )     
        x = self.model.model.cnn.model(x_packed.data) 
        x = x_packed._replace(data=x)                
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(
            x, 
            batch_first=True, 
            padding_value=0.0,
            total_length=int(n_wins.max().item()))

        x, n_wins = self.model.model.time_dependency(x, n_wins)
        x, n_wins = self.model.model.time_dependency_2(x, n_wins)
        x = self.model.model.pool(x, n_wins)
        return x
        
    ### override
    @staticmethod
    def get_librosa_melspec(
        y,
        sr,
        n_fft=1024, 
        hop_length=80, 
        win_length=170,
        n_mels=32,
        fmax=16e3,
        ms_channel=None,
        ):
        '''
        Calculate mel-spectrograms with Librosa.
        '''    
        
        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            S=None,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window='hann',
            center=True,
            pad_mode='reflect',
            power=1.0,
        
            n_mels=n_mels,
            fmin=0.0,
            fmax=fmax,
            htk=False,
            norm='slaney',
            )

        spec = librosa.core.amplitude_to_db(S, ref=1.0, amin=1e-4, top_db=80.0)
        return spec
