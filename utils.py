import numpy as np
import scipy.stats
from scipy.signal import butter, sosfilt
import torch
import os
import torchaudio
from pesq import pesq
from pystoi import stoi


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
    directory = os.path.dirname(file_path)
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


# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Original copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based Loss modules."""

import torch
import torch.nn.functional as F


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True)
    x_stft = torch.view_as_real(x_stft)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss



def log_spectral_distance(
    input,
    target,
    p=2,
    db=True,
    n_fft=400,
    hop_length=160,
    eps=1e-7,
    win_length=None,
    window=None,
    pad=0,
    scale_invariant=False,
    **stft_kwargs,
):
    """
    Implementation of the Log-Spectral Distance (LSD) metric.

    See:
    Gray and Markel, "Distance measures for speech processing," 1976.
    or
    https://en.wikipedia.org/wiki/Log-spectral_distance

    Parameters
    ----------
    input : torch.Tensor
        The input signal. Shape: [..., T]
    target : torch.Tensor
        The target signal. Shape: [..., T]
    p : float
        The norm to use. Default is 2.
    db : bool
        If True, the metric is computed in decibel units,
        i.e., `10.0 * torch.log10` is used instead of `torch.log`.
    n_fft : int
        The number of FFT points. Default is 400.
    hop_length : int
        The hop length for the STFT. Default is 160.
    eps : float
        A small value to avoid numerical instabilities. Default is 1e-7.
    win_length : int or None
        The window length. Default is None, which is equal to n_fft.
    window : torch.Tensor or None
        The window function. Default is None, which is a Hann window.
    pad : int
        The amount of padding to add to the input signal before computing the
        STFT. Default is 0.
    scale_invariant : bool
        If True, the input is rescaled by orthogonal projection of the target onto
        the input sub-space. Default is False.
    stft_kwargs : dict
        Additional arguments to pass to `torchaudio.functional.spectrogram`.

    Returns
    -------
    torch.Tensor
        The log-spectral distance. Shape: [...]
    """
    if win_length is None:
        win_length = n_fft

    if window is None:
        window = torch.hann_window(
            n_fft, periodic=True, dtype=input.dtype, device=input.device
        )

    if p is None or p <= 0:
        raise ValueError(f"p must be a positive number, but got p={p}")

    if scale_invariant:
        scaling_factor = torch.sum(input * target, -1, keepdim=True) / (
            torch.sum(input**2, -1, keepdim=True) + eps
        )
    else:
        scaling_factor = 1.0

    # input log-power spectrum
    # waveform, pad, window, n_fft, hop_length, win_length, power, normalized
    input = torchaudio.functional.spectrogram(
        input,
        pad=pad,
        win_length=win_length,
        window=window,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2,  # power!
        normalized="window",  # keep the level consistent with time-domain level
        **stft_kwargs,
    )
    if db:
        input = 10 * torch.log10(input + eps)
    else:
        input = torch.log(input + eps)

    # target log-power spectrum
    target = torchaudio.functional.spectrogram(
        scaling_factor * target,
        pad=pad,
        win_length=win_length,
        window=window,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2,
        normalized="window",  # keep the level consistent with time-domain level
        **stft_kwargs,
    )
    if db:
        target = 10.0 * torch.log10(target + eps)
    else:
        target = torch.log(target + eps)

    # norm
    denom = (target.shape[-1] * target.shape[-2]) ** (1 / p)  # to get an average value
    lsd = torch.norm(input - target, p=p, dim=(-2, -1)) / denom

    return lsd


class LogSpectralDistance(torch.nn.Module):
    """
    A torch module wrapper for the log-spectral distance.

    See `log_spectral_distance` for more details on the input arguments.
    """
    def __init__(
        self,
        p=2,
        db=True,
        n_fft=400,
        hop_length=160,
        win_length=None,
        window=None,
        pad=0,
        eps=1e-5,
        reduction="mean",
        scale_invariant=False,
        **stft_kwargs,
    ):
        super(LogSpectralDistance, self).__init__()
        self.p = p
        self.eps = eps
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.pad = pad
        self.stft_kwargs = stft_kwargs
        self.db = db
        self.reduction = reduction
        self.scale_invariant = scale_invariant
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("Only reduction=mean|sum|none are supported")

        if window is None:
            window = torch.hann_window(self.win_length, periodic=True)
        self.register_buffer("window", window)

    def forward(self, input, target):
        dist = log_spectral_distance(
            input,
            target,
            p=self.p,
            db=self.db,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            pad=self.pad,
            window=self.window,
            win_length=self.win_length,
            eps=self.eps,
            scale_invariant=self.scale_invariant,
            **self.stft_kwargs,
        )

        if self.reduction == "mean":
            return dist.mean()
        elif self.reduction == "sum":
            return dist.sum()
        else:
            return dist
        



class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window", factor_sc=0.1, factor_mag=0.1):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            factor (float): a balancing factor across different losses.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return self.factor_sc*sc_loss, self.factor_mag*mag_loss

