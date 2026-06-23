import audiomentations as am
import numpy as np
import random
import os
import tempfile
import subprocess
import soundfile as sf
import uuid
import librosa
from os.path import join
from audiomentations.core.transforms_interface import BaseWaveformTransform
import scipy.signal as ss
import scipy.signal

# add root path to sys path
import sys
sys.path.append('/export/home/jrichter/repos/sig-challenge-2023')

from audiomentations.core.utils import (
    find_audio_files_in_paths,
)

from sgmse.util.audiolib import normalize_segmental_rms, active_rms_single, active_rms, is_clipped

EPS = np.finfo(float).eps

def read_file_list(path):            
    file_list = []
    # open file and read the content in a list
    with open(path, 'r') as f:
        for line in f:
            x = line[:-1] # remove linebreak
            file_list.append(x)
    return file_list

# TODO: write custom augmentation for cropped IRs
class ApplyCroppedImpulseResponse(am.ApplyImpulseResponse):
    pass


class Codec(BaseWaveformTransform):

    supports_multichannel = False
    codec_info = {
        'amr-nb' : {
            'bitrates' : {
                #4750 : 'MR475',
                #5150 : 'MR515', 
                #5900 : 'MR59',
                6700 : 'MR67',
                7400 : 'MR74',
                7950 : 'MR795',
                10200 : 'MR102',
                12200 : 'MR122'
            },
            'enc_opts' : '',
            'dec_opts' : '',
            'fs' : 8000,
            'ext' : '',
            'enc_path' : '/export/home/jrichter/repos/sig-challenge-2023/sgmse/util/codecs/amr-nb/encoder',
            'dec_path' : '/export/home/jrichter/repos/sig-challenge-2023/sgmse/util/codecs/amr-nb/decoder'
        },
        'amr-wb' : {
            'bitrates': {
                6600 : '0',
                8850 : '1',
                12650 : '2',
                14250 : '3',
                15850 : '4',
                #18250 : '5',
                #19850 : '6',
                #23050 : '7',
                #23850 : '8'
            },
            'enc_opts' : '',
            'dec_opts' : '',
            'fs' : 16000,
            'ext' : '',
            'enc_path' : '/export/home/jrichter/repos/sig-challenge-2023/sgmse/util/codecs/amr-wb/encoder',
            'dec_path' : '/export/home/jrichter/repos/sig-challenge-2023/sgmse/util/codecs/amr-wb/decoder'
        },
        'evs' : {
            'bitrates' : { # this includes the sampling rate in the string
                #5900 : '5900 48',
                7200 : '7200 48',
                8000 : '8000 48',
                9600 : '9600 48',
                13200 : '13200 48',
                24400 : '24400 48',
                #32000 : '32000 48',
                #48000 : '48000 48',
                #64000 : '64000 48',
                #96000 : '96000 48', 
                #128000 : '128000 48'
            },
            'enc_opts' : '-q',
            'dec_opts' : '48',
            'fs' : 48000,
            'ext' : '',
            'enc_path' : '/export/home/jrichter/repos/sig-challenge-2023/sgmse/util/codecs/evs/encoder',
            'dec_path' : '/export/home/jrichter/repos/sig-challenge-2023/sgmse/util/codecs/evs/decoder'
        },
        'codec2' : {
            'bitrates' : {
                #450 : '450',
                #1200 : '1200',
                #1300 : '1300',
                #1400 : '1400',
                1600 : '1600',
                2400 : '2400',
                3200 : '3200'
            },
            'enc_opts' : '',
            'dec_opts' : '1300',  # Workaround for a small bug in codec2
            'fs' : 8000,
            'ext' : '.c2',
            'enc_path' : '/export/home/jrichter/repos/sig-challenge-2023/sgmse/util/codecs/codec2/encoder',
            'dec_path' : '/export/home/jrichter/repos/sig-challenge-2023/sgmse/util/codecs/codec2/decoder'
        }
    }


    # def __init__(self, use_codecs=['amr-nb', 'amr-wb', 'evs', 'codec2'], bitrate : int = None, p : float = 0.5):
    def __init__(self, use_codecs=['amr-nb', 'amr-wb', 'evs'], bitrate : int = None, p : float = 0.5):

        super().__init__(p)
        
        for codec in use_codecs:
            if codec not in self.codec_info:
                raise ValueError

        self.use_codecs = use_codecs
        self.tempdir = tempfile.TemporaryDirectory()
        
    def __del__(self):
        self.tempdir.cleanup()
        
    def randomize_parameters(self, samples: np.ndarray, sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        
        if self.parameters["should_apply"]:
            #codec_name = random.choice(list(self.codec_info.keys()))
            codec_name = random.choice(self.use_codecs)
            codec = self.codec_info[codec_name]
            self.parameters["bitrate"] = random.choice(list(codec["bitrates"].keys()))            
            self.parameters["codec"] = codec
            
    def apply(self, samples: np.ndarray, sample_rate: int):        
        codec = self.parameters["codec"]
        bitrate = self.parameters["bitrate"]
        sample_rate_codec = codec["fs"]
        
        # define some file names 
        file_uuid = uuid.uuid4()
        file_orig = f'{self.tempdir.name}/{file_uuid}_orig.wav'
        file_orig_raw_resamp = f'{self.tempdir.name}/{file_uuid}_orig_resamp.raw'
        file_encoded = f'{self.tempdir.name}/{file_uuid}_encoded{codec["ext"]}'
        file_decoded = f'{self.tempdir.name}/{file_uuid}_decoded.raw'
        file_decoded_final = f'{self.tempdir.name}/{file_uuid}_decoded_final.wav'
        
        # write given samples
        sf.write(file_orig, samples, samplerate=sample_rate)
        
        # convert to raw format and resample with sox
        subprocess.run(['sox', 
                        file_orig, 
                        '-t', 'raw', 
                        '-e', 'signed-integer', 
                        '-b', '16',
                        '-r', str(sample_rate_codec),
                        '-G',
                        file_orig_raw_resamp
                        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # run encoder
        subprocess.run(f"{codec['enc_path']} {codec['enc_opts']} {codec['bitrates'][bitrate]} {file_orig_raw_resamp} {file_encoded}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  
        # run decoder
        #print(f"{codec['dec_path']} {codec['dec_opts']} {file_encoded} {file_decoded}", )
        subprocess.run(f"{codec['dec_path']} {codec['dec_opts']} {file_encoded} {file_decoded}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # convert back to wav and resample with sox
        subprocess.run(['sox', 
                '-t', 'raw', 
                '-e', 'signed-integer', 
                '-b', '16',
                '-r', str(sample_rate_codec),
                file_decoded,
                '-r', str(sample_rate),
                '-G',
                file_decoded_final
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # read final wav back into numpy
        new_samples, _ = librosa.load(file_decoded_final, sr=None)
        
        # remove files
        os.remove(file_orig)
        os.remove(file_orig_raw_resamp)
        os.remove(file_encoded)
        os.remove(file_decoded)
        os.remove(file_decoded_final)

        # pad or cut to original length
        len_diff = samples.shape[-1] - new_samples.shape[-1]
        if len_diff > 0:
            new_samples = np.pad(y, (0, len_diff), mode='constant')
        elif len_diff < 0:
            new_samples = new_samples[:samples.shape[-1]]

        return new_samples


class DropSamples(BaseWaveformTransform):     
    
    supports_multichannel = False
    
    def __init__(self, packet_width=20, lost_packages_min=1, lost_packages_max=5, drops_per_sec_min=3, drops_per_sec_max=6, p=0.5):
        super().__init__(p)
        
        self.packet_width = packet_width
        self.lost_packages_min = lost_packages_min
        self.lost_packages_max = lost_packages_max
        self.drops_per_sec_min = drops_per_sec_min
        self.drops_per_sec_max = drops_per_sec_max
        
    def randomize_parameters(self, samples: np.ndarray, sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        
        if self.parameters["should_apply"]:
            num_samples = samples.shape[-1]
            len_in_sec = num_samples / sample_rate
            drops_per_sec = random.uniform(self.drops_per_sec_min, self.drops_per_sec_max)
            num_drops = int(len_in_sec * drops_per_sec)
            
            # Generate sample drops of different widths
            indices = []
            for _ in range(num_drops):
                num_lost_packages = random.randint(self.lost_packages_min, self.lost_packages_max)
                width = int(self.packet_width/1000 * num_lost_packages * sample_rate)
                start_idx = random.randint(0, num_samples-width)
                indices = indices + list(range(start_idx, start_idx+width))
            
            self.parameters["indices"] = indices
            
    
    def apply(self, samples: np.ndarray, sample_rate: int):
        new_samples = samples.copy()
        
        indices = self.parameters["indices"]
        new_samples[indices] = 0.0
        return new_samples    
        
    
class DropSamplesDepricated(BaseWaveformTransform):
    
    supports_multichannel = False
    
    def __init__(self, min_width : int = 2, max_width : int = 100, drops_per_sec_min : int = 5, drops_per_sec_max : int = 20, p : float = 0.5):
        super().__init__(p)
        
        if min_width < 0 or max_width < 0:
            raise ValueError("min_width and max_width must be non-negative")
        if max_width < min_width:
            raise ValueError("max_width must be greater or equal to min_width")
        if drops_per_sec_min < 0 or drops_per_sec_max < 0:
            raise ValueError("drops_per_sec_min/max must be non-negative")
        if drops_per_sec_max < drops_per_sec_min:
            raise ValueError("drops_per_sec_max must greater or equal to drops_per_sec_min")
        
        self.min_width = min_width
        self.max_width = max_width
        self.drops_per_sec_min = drops_per_sec_min
        self.drops_per_sec_max = drops_per_sec_max
        
    def randomize_parameters(self, samples: np.ndarray, sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        
        if self.parameters["should_apply"]:
            num_samples = samples.shape[-1]
            len_in_sec = num_samples / sample_rate
            drops_per_sec = random.uniform(self.drops_per_sec_min, self.drops_per_sec_max)
            num_drops = int(len_in_sec * drops_per_sec)
            
            # Generate sample drops of different widths
            indices = []
            for _ in range(num_drops):
                start_idx = random.randint(0, num_samples-self.max_width)
                width = random.randint(self.min_width, self.max_width)
                indices = indices + list(range(start_idx, start_idx+width))
            
            self.parameters["indices"] = indices
            
    
    def apply(self, samples: np.ndarray, sample_rate: int):
        new_samples = samples.copy()
        
        indices = self.parameters["indices"]
        new_samples[indices] = 0.0
        return new_samples


class BandwidthReduction(BaseWaveformTransform):
    
    supports_multichannel = False
    
    def __init__(self, scale_factors=[2, 4, 8], lp_types=["bessel", "butter", "cheby2"], lp_orders=[2, 4, 8], bwe_method="decimate", p=0.5):
        super().__init__(p)

        self.scale_factors = scale_factors
        self.lp_types = lp_types
        self.lp_orders = lp_orders
        self.bwe_method = bwe_method
        
        
    def randomize_parameters(self, samples: np.ndarray, sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        
        if self.parameters["should_apply"]:
            self.parameters["scale_factor"] = random.choice(self.scale_factors)
            self.parameters["lp_type"] = random.choice(self.lp_types)
            self.parameters["lp_order"] = random.choice(self.lp_orders)
    
    def apply(self, samples: np.ndarray, sample_rate: int):
        lossy_speech = samples.copy()

        Wn = 1./(2*self.parameters["scale_factor"])
        if self.parameters["lp_type"] == "bessel":
            kwargs = {}
        elif self.parameters["lp_type"] == "butter":
            kwargs = {}
        elif self.parameters["lp_type"] == "cheby2": 
            kwargs = {"rs": 10. + 20*np.random.random()}

        if self.parameters["lp_order"] > 2:
            kwargs["output"] = "sos"
            
        lp_filter_coefs = getattr(scipy.signal, self.parameters["lp_type"])(N=self.parameters["lp_order"], Wn=Wn, fs=1, **kwargs)

        if self.bwe_method == "decimate": #method used by HiFI++ and VoiceFixer
            z, p, k = ss.sos2zpk(lp_filter_coefs) if self.parameters["lp_order"] > 2 else ss.tf2zpk(*lp_filter_coefs)
            filter_instance = ss.dlti(z,p,k)
            lossy_speech_subsampled = ss.decimate(lossy_speech, q=self.parameters["scale_factor"], ftype=filter_instance)
            lossy_speech = ss.resample_poly(lossy_speech_subsampled, up=self.parameters["scale_factor"], down=1)
        elif self.bwe_method == "polyphase": #method used by NVSR
            sos = lp_filter_coefs if self.parameters["lp_order"] > 2 else ss.tf2sos(*lp_filter_coefs)
            lossy_speech_filtered = ss.sosfilt(sos, lossy_speech)
            lossy_speech_subsampled = ss.resample_poly(lossy_speech_filtered, down=self.parameters["scale_factor"], up=1)
            lossy_speech = ss.resample_poly(lossy_speech_subsampled, up=sample_rate, down=sample_rate/self.parameters["scale_factor"])
        
        return lossy_speech
    
    
class BandwidthReductionCodecs(BaseWaveformTransform):
    
    def __init__(self, p=0.5):
        super().__init__(p)
        
        self.bwr_aug = BandwidthReduction(scale_factors=[2, 4, 8], lp_types=["bessel", "butter", "cheby2"], lp_orders=[2, 4, 8], bwe_method="decimate", p=p)
        self.mp3_aug = am.Mp3Compression(min_bitrate=8, max_bitrate=64, p=p)
        self.codec_aug = Codec(p=p)
        
    def randomize_parameters(self, samples: np.ndarray, sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        
        self.bwr_aug.randomize_parameters(samples, sample_rate)
        self.mp3_aug.randomize_parameters(samples, sample_rate)
        self.codec_aug.randomize_parameters(samples, sample_rate)
        
    def apply(self, samples: np.ndarray, sample_rate: int):
        
        index = random.randint(0, 2)
        if index == 0:
            return self.bwr_aug(samples, sample_rate)
        elif index == 1:
            return self.mp3_aug(samples, sample_rate)
        elif index == 2:
            return self.codec_aug(samples, sample_rate)
    
    
class AddBackgroundNoiseSegmentalSNR(BaseWaveformTransform):
    def __init__(self, root_dir, noise_list, min_snr_in_db=-10, max_snr_in_db=15, p=0.5, return_components=False):
        super().__init__(p)
        
        self.root_dir = root_dir
        self.noise_list = noise_list
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        self.noise_file_paths = read_file_list(noise_list)
        self.return_components = return_components
        
        
    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["snr_in_db"] = random.uniform(
                self.min_snr_in_db, self.max_snr_in_db
            )
            # sample ramdom noise file and remove it from list
            elem = random.choice(self.noise_file_paths)
            self.parameters["noise_file_path"] = elem
            self.noise_file_paths.remove(elem)
            if len(self.noise_file_paths) == 0:
                self.noise_file_paths = read_file_list(self.noise_list)
            
            
    def apply(self, samples: np.ndarray, sample_rate: int):
        clean = samples.copy()
        noise_path = join(self.root_dir, self.parameters["noise_file_path"])
        noise, _ = sf.read(noise_path)
        
         # Repeat the sound if it shorter than the input sound
        num_samples = len(clean)
        while len(noise) < num_samples:
            noise = np.concatenate((noise, noise))

        if len(noise) > num_samples:
            noise = noise[0:num_samples]

        clean = clean/(max(abs(clean))+EPS)
        noise = noise/(max(abs(noise))+EPS)   
        rmsclean, rmsnoise = active_rms(clean=clean, noise=noise)
        clean = normalize_segmental_rms(clean, rms=rmsclean, target_level=-25)
        noise = normalize_segmental_rms(noise, rms=rmsnoise, target_level=-25)
        
        snr = self.parameters["snr_in_db"]
        noisescalar = rmsclean / (10**(snr/20)) / (rmsnoise+EPS)
        noise = noise * noisescalar
        
        mixture = clean + noise
        
        clipping_threshold = 0.99
        if is_clipped(mixture):
            maxamplevel = max(abs(mixture))/(clipping_threshold-EPS)
            mixture = mixture/maxamplevel
            clean = clean/maxamplevel
        
        if self.return_components:
            return mixture, clean, snr, noise_path
        else:      
            return mixture
    

 
 
 # test codec aug
if __name__ == '__main__':
    # aug = am.Mp3Compression(min_bitrate=8, max_bitrate=64, p=1)
    # aug = Codec(p=1)

    # aug = AddBackgroundNoiseSegmentalSNR(root_dir="/data3/databases/DNS4/datasets_fullband/noise_fullband",  noise_list="/export/home/jrichter/repos/sig-challenge-2023/config/noisefiles_valid_debug.txt", min_snr_in_db=3.1819, max_snr_in_db=3.182, p=1)
    # clipping_aug = am.ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=60, p=1)
    
    aug = BandwidthReductionCodecs(p=1)
    
    aug = am.Gain(min_gain_in_db=6, max_gain_in_db=6, p=1.0)
    
    s, fs = sf.read('/data3/databases/MultiDist/test/clean/book_01187_chp_0011_reader_02383_1_seg_1.wav')
    # s, fs = sf.read('/data3/databases/DNS4/datasets_fullband/clean_fullband/vctk_wav48_silence_trimmed/p287/p287_424_mic1.wav')
    sf.write(f's.wav', s, samplerate=fs)
    
    for i in range(1):
        x = aug(s, fs)
        sf.write(f'x_{i}.wav', x, samplerate=fs)
    
    