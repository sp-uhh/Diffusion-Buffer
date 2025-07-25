
from os.path import join
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
from torchaudio import load
import torchaudio
import audiomentations as am
import numpy as np
import torch.nn.functional as F


def get_window(window_type, window_length):
    if window_type == 'sqrthann':
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    elif window_type == 'hann':
        return torch.hann_window(window_length, periodic=True)
    else:
        raise NotImplementedError(f"Window type {window_type} not implemented!")


class Specs(Dataset):
    def __init__(self, data_dir, subset, dummy, shuffle_spec, num_frames,
            format='noise', normalize="noisy", spec_transform=None,
            stft_kwargs=None, gains = [0, 0], fs=16000, **ignored_kwargs):

        subdir = join(data_dir, subset)
        
        # Read file paths according to file naming format.
        if format == "default":
            self.clean_files = sorted(glob(join(data_dir, subset, "clean", "*.wav")))
            self.noisy_files = sorted(glob(join(data_dir, subset, "noisy", "*.wav")))
        elif format == "ears_wham":
            self.clean_files = sorted(glob(join(data_dir, subset, "clean", "**", "*.wav"), recursive=True))
            self.noisy_files = sorted(glob(join(data_dir, subset, "noisy", "**", "*.wav"), recursive=True))
        else:
            # Feel free to add your own directory format
            raise NotImplementedError(f"Directory format {format} unknown!")

        
        self.num_utterances = len(self.clean_files)
        self.dummy = dummy
        self.num_frames = num_frames
        self.shuffle_spec = shuffle_spec
        self.normalize = normalize
        self.gains = gains
        self.fs = fs
        self.spec_transform = spec_transform
        if self.gains is not None and not self.gains == [0, 0]:
            self.gain_augmentation = am.Gain(min_gain_in_db=self.gains[0], max_gain_in_db=self.gains[1], p=1.0)


        assert all(k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]), "misconfigured STFT kwargs"
        self.stft_kwargs = stft_kwargs
        self.hop_length = self.stft_kwargs["hop_length"]
        assert self.stft_kwargs.get("center", None) == True, "'center' must be True for current implementation"
        self.target_len = (self.num_frames - 1) * self.hop_length

    def __getitem__(self, i):
        x, ori_sr = load(self.clean_files[i])
        y, ori_sr = load(self.noisy_files[i])
        
        if x.shape != y.shape:
            _min = np.min([x.shape[-1], y.shape[-1]])
            x = x[:, :_min]
            y = y[:, :_min]
  
        if ori_sr != self.fs:
            x = torchaudio.functional.resample(x, ori_sr, self.fs)
            y = torchaudio.functional.resample(y, ori_sr, self.fs)

        # formula applies for center=True

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
            normfac = y.abs().max()
        elif self.normalize == "clean":
            normfac = x.abs().max()
        elif self.normalize == "not":
            normfac = 1.0
        x = x / (normfac + 1e-6)
        y = y / (normfac + 1e-6)
        
        
        # Apply random gain (if `gains` was passed at init)
        if self.gains is not None and not self.gains == [0, 0]:
            y = self.gain_augmentation(y, sample_rate=None)
            self.gain_augmentation.freeze_parameters()
            x = self.gain_augmentation(x, sample_rate=None)  # sample_rate must be passed, but is unused
            self.gain_augmentation.unfreeze_parameters()

        X = torch.stft(x, **self.stft_kwargs)
        Y = torch.stft(y, **self.stft_kwargs)
        X, Y = self.spec_transform(X), self.spec_transform(Y)    
        
    
        return X, Y

    def __len__(self):
        if self.dummy:
            # for debugging shrink the data set size
            return int(len(self.clean_files)/200)
        else:
            return len(self.clean_files)



    
    

class SpecsDataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--base_dir", type=str, required=True, help="The base directory of the dataset. Should contain `train`, `valid` and `test` subdirectories, each of which contain `clean` and `noisy` subdirectories.")
        parser.add_argument("--format", type=str, default="default", help="Read file paths according to file naming format.")
        parser.add_argument("--batch_size", type=int, default=8, help="The batch size. 8 by default.")
        parser.add_argument("--n_fft", type=int, default=510, help="Number of FFT bins. 510 by default.")   # to assure 256 freq bins
        parser.add_argument("--hop_length", type=int, default=128, help="Window hop length. 128 by default.")
        parser.add_argument("--num_frames", type=int, default=256, help="Number of frames for the dataset. 256 by default.")
        parser.add_argument("--window", type=str, choices=("sqrthann", "hann"), default="hann", help="The window function to use for the STFT. 'hann' by default.")
        parser.add_argument("--num_workers", type=int, default=4, help="Number of workers to use for DataLoaders. 4 by default.")
        parser.add_argument("--dummy", action="store_true", help="Use reduced dummy dataset for prototyping.")
        parser.add_argument("--spec_factor", type=float, default=0.15, help="Factor to multiply complex STFT coefficients by. 0.15 by default.")
        parser.add_argument("--spec_abs_exponent", type=float, default=0.5, help="Exponent e for the transformation abs(z)**e * exp(1j*angle(z)). 0.5 by default.")
        parser.add_argument("--normalize", type=str, choices=("clean", "noisy", "not"), default="noisy", help="Normalize the input waveforms by the clean signal, the noisy signal, or not at all.")
        parser.add_argument("--transform_type", type=str, choices=("exponent", "log", "none"), default="exponent", help="Spectogram transformation for input representation.")
        parser.add_argument("--fs", type=int, default=16000, 
                            help="Define the sample rate.")
        parser.add_argument("--gains", type=int, nargs=2, required=False, default=[0,0],
                            help="If passed, samples a gain value g to apply to both x0 and y (from a uniform distribution with the given bounds in dB, for each data point and call). NOTE: not really a correct model for y=A(g*x0) when the corruption A is nonlinear. We'll use it anyways for training speed")

        return parser

    def __init__(
        self, base_dir, format='default', batch_size=8,
        n_fft=510, hop_length=128, num_frames=256, window='hann',
        num_workers=4, dummy=False, spec_factor=0.15, spec_abs_exponent=0.5,
        gpu=True, normalize='noisy', transform_type="exponent", fs=16000, gains=[-12,6], **kwargs
    ):
        super().__init__()
        self.base_dir = base_dir
        self.format = format
        self.batch_size = batch_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.gains = gains
        self.window = get_window(window, self.n_fft)
        self.windows = {}
        self.num_workers = num_workers
        self.dummy = dummy
        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
        self.gpu = gpu
        self.normalize = normalize
        self.transform_type = transform_type
        self.fs = fs
        self.target_len = (self.num_frames - 1) * self.hop_length
        self.kwargs = kwargs

    def setup(self, stage=None):
        specs_kwargs = dict(
            stft_kwargs=self.stft_kwargs, num_frames=self.num_frames,
            spec_transform=self.spec_fwd, **self.kwargs
        )
        if stage == 'fit' or stage is None:
            self.train_set = Specs(data_dir=self.base_dir, subset='train',
                dummy=self.dummy, shuffle_spec=True, format=self.format, 
                normalize=self.normalize, gains = self.gains, fs=self.fs, **specs_kwargs)
            self.valid_set = Specs(data_dir=self.base_dir, subset='valid',
                dummy=self.dummy, shuffle_spec=False, format=self.format,
                normalize=self.normalize, gains = self.gains, fs=self.fs, **specs_kwargs)
        if stage == 'test' or stage is None:
            self.test_set = Specs(data_dir=self.base_dir, subset='test',
                dummy=self.dummy, shuffle_spec=False, format=self.format,
                normalize=self.normalize, gains = self.gains, fs=self.fs, **specs_kwargs)

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
