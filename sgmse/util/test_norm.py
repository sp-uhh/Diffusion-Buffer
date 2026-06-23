import numpy as np 
import os
from os.path import join
import glob
import torch, torchaudio

norms = [[], []]

path = "/data3/databases/VB-SIG-32k/valid"
for f, g in zip( sorted(glob.glob(join(path, "noisy/*.wav"))), sorted(glob.glob(join(path, "clean/*.wav"))) ):
    y, _ = torchaudio.load(f)
    x, _ = torchaudio.load(g)

    norms[0].append(y.abs().max())
    norms[1].append(x.abs().max())

norms = np.array(norms)
print(f"Mean amplitude of y: {np.mean(norms[0]):.3f} Mean amplitude of x: {np.mean(norms[1]):.3f}") 
print(f"Amplitude std of y: {np.std(norms[0]):.3f} Amplitude std of x: {np.std(norms[1]):.3f}") 
print(f"Amplitude max of y: {np.max(norms[0]):.3f} Amplitude max of x: {np.max(norms[1]):.3f}") 
print(f"Amplitude min of y: {np.min(norms[0]):.3f} Amplitude min of x: {np.min(norms[1]):.3f}") 
# print(norms[0][norms[0]<0.1])
print(f"{(len(norms[0][norms[0]<0.1])/len(norms[0]) * 100):.1f} % of y amplitudes are lower than 0.1")


