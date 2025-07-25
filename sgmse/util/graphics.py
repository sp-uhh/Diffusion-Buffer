import torch
from torchaudio import load
import matplotlib.pyplot as plt
import numpy as np
import os
import soundfile as sf

# Plotting settings
EPS_graphics = 1e-10
n_fft = 512
hop_length = 128
vmin, vmax = -60, 0

stft_kwargs = {"n_fft": n_fft, "hop_length": hop_length, "window": torch.hann_window(n_fft), "center": True, "return_complex": True}

def visualize_example(mix, estimate, target, idx_sample=0, epoch=0, name="", sample_rate=16000, hop_len=128, return_fig=False):
	"""Visualize training targets and estimates of the Neural Network
	Args:
		- mix: Tensor [F, T]
		- estimates/targets: Tensor [F, T]
	"""

	if isinstance(mix, torch.Tensor):
		mix = torch.abs(mix).detach().cpu()
		estimate = torch.abs(estimate).detach().cpu()
		target = torch.abs(target).detach().cpu()

	fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 8))

	freqs = sample_rate/(2*mix.size(-2)) * torch.arange(mix.size(-2))
	frames = hop_len/sample_rate * torch.arange(mix.size(-1))

	ax = axes.flat[0]
	im = ax.pcolormesh(frames, freqs, 20*np.log10(.1*mix + EPS_graphics), vmin=vmin, vmax=vmax, shading="auto", cmap="magma")
	ax.set_xlabel('Time [s]')
	ax.set_ylabel('Frequency [Hz]')
	ax.set_title('Mixture')

	ax = axes.flat[1]
	ax.pcolormesh(frames, freqs, 20*np.log10(.1*estimate + EPS_graphics), vmin=vmin, vmax=vmax, shading="auto", cmap="magma")
	ax.set_xlabel('Time [s]')
	ax.set_ylabel('Frequency [Hz]')
	ax.set_title('Estimate')

	ax = axes.flat[2]
	ax.pcolormesh(frames, freqs, 20*np.log10(.1*target + EPS_graphics), vmin=vmin, vmax=vmax, shading="auto", cmap="magma")
	ax.set_xlabel('Time [s]')
	ax.set_ylabel('Frequency [Hz]')
	ax.set_title('Clean')

	fig.subplots_adjust(right=0.87)
	cbar_ax = fig.add_axes([0.9, 0.25, 0.005, 0.5])
	fig.colorbar(im, cax=cbar_ax)

	if return_fig:
		return fig
	else:
		plt.savefig(os.path.join(f"spectro_{idx_sample}_epoch{epoch}{name}.png"), bbox_inches="tight")
		plt.close()
