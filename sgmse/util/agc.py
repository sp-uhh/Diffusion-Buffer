import numpy as np
import torch, torchaudio
import os
from os.path import join
import tqdm
import glob
import matplotlib.pyplot as plt
from pedalboard import Pedalboard, Compressor, Limiter
import matplotlib.pyplot as plt

INIT_BUFFER_SIZE = .1
NORMALIZATION_RISE_BUFFER_SIZE = .1
VAD_THRESHOLD = 1e-3
VAD_BUFFER_SIZE_S = .1
RISE_MODE = "exp"
# STOP_TRACKING = 3.
STOP_TRACKING = 1e8
ISTFT_KWARGS = {"n_fft": 638, "hop_length": 160, "center": True, "window": torch.hann_window(638)}
STFT_KWARGS = {**ISTFT_KWARGS, "return_complex": True}

THRESHOLD_SPP = .8
LP_FILTER_FREQ_HZ = 4000
LOOKAHEAD_S = .02

def seconds_to_stft_frames(t_in_seconds, fs, stft_kwargs):
    return int(t_in_seconds*fs/stft_kwargs["hop_length"])

def seconds_to_time_frames(t_in_seconds, fs, **ignored_kwargs):
    return int(t_in_seconds*fs)

def normalize_online(y, fs, stft_kwargs=None,
    init_buffer_size=INIT_BUFFER_SIZE,
    normalization_rise_buffer_size=NORMALIZATION_RISE_BUFFER_SIZE,
    vad_buffer_size_s=VAD_BUFFER_SIZE_S,
    rise_mode=RISE_MODE,
    vad_threshold=VAD_THRESHOLD,
    stop_tracking=STOP_TRACKING,
    threshold_spp=THRESHOLD_SPP,
    lp_filter_freq_hz=LP_FILTER_FREQ_HZ,
    lookahead_s=LOOKAHEAD_S
    ):

    s2f = seconds_to_stft_frames if stft_kwargs is not None else seconds_to_time_frames #Counting system

    # vad_buffer = [False]*s2f(vad_buffer_size, fs=fs, stft_kwargs=stft_kwargs)

    start_tracking = False
    tracked_max = vad_threshold*torch.ones_like(y.abs()[0]) #T
    first_rise = True
    previous_max = 1.

    start_normalize = False

    normalization_rise_step = 0
    end_normalization_rise = False

    ### VAD
    vad_estimator = VADEstimator(
        frame_length=stft_kwargs["n_fft"],
        threshold_spp=threshold_spp,
        fs=fs,
        hop_length=stft_kwargs["hop_length"],
        lp_filter_freq_hz=lp_filter_freq_hz,
        vad_buffer_size_s=vad_buffer_size_s,
        lookahead_s=lookahead_s
    )
    spp, vad_index = vad_estimator.process(y)

    # fig, axes = plt.subplots(3, 1)
    # ax = axes.flatten()[0]
    # ax.imshow(20*np.log10(y.abs() + 1e-10), origin="lower")
    # ax = axes.flatten()[1]
    # ax.imshow(20*np.log10(spp + 1e-10), origin="lower")
    # ax.axvline(x=vad_index, color="r")
    # ax = axes.flatten()[2]
    # ax.plot(np.mean(spp, axis=0), color="b")
    # ax.plot(np.mean(spp[: 40, :], axis=0), color="k")
    # ax.axvline(x=vad_index, color="r")
    # ax.set_xlim([0, y.shape[-1]])
    # plt.savefig(join("./.tests/vad/post_process",  f"{y.size(-1)}.png"))

    for t in range(y.size(-1)):
        ### VAD
        # vad = (y[..., t].abs().mean().item() > vad_threshold)
        # vad_buffer = vad_buffer[1: ] + [vad]
        # if np.array(vad_buffer).all() and not start_tracking:
        #     start_tracking = True
        if t == vad_index:
            start_tracking = True

        ### MAX TRACKING
        if start_tracking and y[..., t].abs().mean() > tracked_max[t-1] and t < s2f(stop_tracking, fs=fs, stft_kwargs=stft_kwargs):
            tracked_max[t] = min(1., y[..., t].abs().mean())
        else:
            tracked_max[t] = tracked_max[t-1] if t else vad_threshold

        ### NORMALIZING
        if start_tracking and not start_normalize and t > s2f(init_buffer_size, fs=fs, stft_kwargs=stft_kwargs):
            start_normalize = True

        if start_normalize:
            if tracked_max[t] > tracked_max[t-1]: #We have discovered a new maximum, don't jump into it, but re-do a rise:
                end_normalization_rise = False
                normalization_rise_step = 0
                if not first_rise:
                    previous_max = tracked_max[t-1]

            if end_normalization_rise: #Stable regime: we are not in a rise
                y[..., t] /= tracked_max[t]
            else: #Transition regime: we are in a rise because either leaving the initialization buffer, or having discovered a new max
                normalization_rise_step += 1/s2f(normalization_rise_buffer_size, fs=fs, stft_kwargs=stft_kwargs)
                if rise_mode == "lin":
                    normalization_rise_value = ( tracked_max[t] * normalization_rise_step ) + ( previous_max * (1-normalization_rise_step) ) 
                elif rise_mode == "exp":
                    normalization_rise_value = ( tracked_max[t]**normalization_rise_step ) * ( previous_max ** (1-normalization_rise_step) )
                y[..., t] /= normalization_rise_value

            if normalization_rise_step > 1 and not end_normalization_rise: #We have achieved the rise, state its end and inspect whether that was the first
                end_normalization_rise = True
                if first_rise:
                    first_rise = False

    return y

# THRESHOLD_CLIP = 1.
# THRESHOLD_ACT = .96
# COEF_TANH = 5

# def declip_online(y,
#     threshold_act=THRESHOLD_ACT,
#     threshold_clip=THRESHOLD_CLIP,
#     ):


#     t_act = np.arange(y.size(-1))[y.abs()>threshold_act]

#     plt.figure()
#     # plt.plot(y[151470: 151490], color="r", marker="+", linestyle="")
#     plt.plot(y[151570: 151600], color="r", marker="+", linestyle="")
#     y[t_act] = torch.sign(y[t_act]) * (threshold_act + (threshold_clip-threshold_act) * torch.tanh( COEF_TANH * (y[t_act].abs() - threshold_act)) )
#     plt.plot(y[151570: 151600], color="k", marker="+", linestyle="")
#     plt.ylim([0.95, 1.05])
#     plt.savefig("./fig_dip1.png")

#     return y


# ATTACK = .1
# RELEASE = .99
# THRESHOLD = 0.9
# DELAY = 50

# class Limiter:
#     def __init__(self, 
#     attack_coeff=ATTACK, 
#     release_coeff=RELEASE, 
#     delay=DELAY, 
#     threshold=THRESHOLD):
#         self.delay_index = 0
#         self.envelope = 0
#         self.gain = 1
#         self.delay = delay
#         self.delay_line = np.zeros(delay)
#         self.release_coeff = release_coeff
#         self.attack_coeff = attack_coeff
#         self.threshold = threshold

#     def limit(self, signal):
#         for idx, sample in enumerate(signal):
#             self.delay_line[self.delay_index] = sample
#             self.delay_index = (self.delay_index + 1) % self.delay

#             # calculate an envelope of the signal
#             self.envelope  = max(abs(sample), self.envelope*self.release_coeff)

#             if self.envelope > self.threshold:
#                 target_gain = self.threshold / self.envelope
#             else:
#                 target_gain = 1.0

#             # have self.gain go towards a desired limiter gain
#             self.gain = ( self.gain*self.attack_coeff +
#                           target_gain*(1-self.attack_coeff) )

#             # limit the delayed signal
#             signal[idx] = self.delay_line[self.delay_index] * self.gain
#         return signal


class CompressorPedalboard():

    def __init__(self):

        self.board = Pedalboard([Compressor(threshold_db=-3, ratio=15)])

    def limit(self, audio, samplerate):
        return self.board(audio, samplerate)


# default values for SPP-based noise PSD estimator
SPP_FIX_SMOOTH = 0.8
SPP_PROB_SMOOTH = 0.9
SPP_PRIOR = 0.5
SPP_SNR_OPT_DB = 15
SPP_NUM_FRAMES_INIT = 10

class SPPNoiseEstimator:

    """
        Implements the speech presence probability (SPP) based noise PSD
        estimator proposed in [1], [2].

        NOTE: This algorithm is designed for 32 ms with 16 ms shift. If you
        want to use other parameters for your STFT, you need to adjust the
        parameters accordingly. Even after adjusting these parameters, a lower
        performance can be expected.

        [1] T. Gerkmann and R. C. Hendriks, “Noise power estimation based on
        the probability of speech presence,” in IEEE Workshop on Applications
        of Signal Processing to Audio and Acoustics (WASPAA), New Paltz, NY,
        USA, 2011, pp. 145–148.
        [2] T. Gerkmann and R. C. Hendriks, “Unbiased MMSE-based noise power
        estimation with low complexity and low tracking delay,” IEEE
        Transactions on Audio, Speech, and Language Processing, vol. 20, no. 4,
        pp. 1383–1393, May 2012.
    """

    def __init__(self, frame_length,
                 fixed_smooth=SPP_FIX_SMOOTH,
                 prob_smooth=SPP_PROB_SMOOTH,
                 prior=SPP_PRIOR,
                 snr_opt_db=SPP_SNR_OPT_DB,
                 num_frames_init=SPP_NUM_FRAMES_INIT):
        """ Generates a SPP noise PSD estimator object. For all optional
        parameters, the default values in [1] are used.

        :frame_length: frame length (number of samples)
        :fixed_smooth: (optional) fixed smoothing constant (see line 6 of
                       Algorithm 1 in [1]). Default value is 0.8.

        :prob_smooth: (optional) fixed smoothing constant for smoothing the
                      SPP values (see line 3 of Algorithm 1 in [1]). Default
                      value is 0.9.
        :prior: (optional) prior probability for speech (P(H1) in [1]). Default
                value is 0.5.
        :snr_opt_db: (optional) fixed speech SNR (dB) (xi_opt in [1]). Default
                value is 15 dB.
        :num_frames_init: (optional) number of frames to be used for
                          initialization. Default value is 10.
        """
        self._frame_length = frame_length

        # fixed smoothing constant for smoothing the noise periodogram
        self._fixed_smooth = fixed_smooth

        # fixed smoothing constant for smoothing the SPP
        self._prob_smooth = prob_smooth

        # prior probability for speech presence (P(H1) in [1])
        self._prior = prior

        # fixed SNR (\xi_opt in [1])
        self._snr_opt_lin = 10.**(snr_opt_db/10.)

        # number of frames used for initialization
        self._num_frames_init = num_frames_init

        # internal states
        self._v_old_psd = np.zeros(frame_length // 2 + 1)
        # self._v_old_psd = self._snr_opt_lin*np.ones(frame_length // 2 + 1)

        self._v_smooth_prob = np.zeros(frame_length // 2 + 1)
        self._inv_glr_factor = (1 - prior)/prior*(1. + self._snr_opt_lin)
        self._inv_glr_exp_factor = self._snr_opt_lin/(1. + self._snr_opt_lin)
        self._num_frames_processed = 0

    def update(self, v_noisy_per):
        """ Estimates noise PSD from the noisy input periodogram.

        The computations correspond to Algorithm 1 in [1]. All computations are
        performed for a single frame. Therefore, all input parameters have to
        be vectors with DFT length // 2 + 1 elements.

        :noisy_per: noisy periodogram (|Y|^2) (numpy array)
        :returns: noise PSD (sigma^2_n) (numpy array)

        """
        if self._num_frames_processed < self._num_frames_init:
            # average first frames to obtain first noise PSD estimate
            v_noise_psd = self._v_old_psd + v_noisy_per / self._num_frames_init

            self._v_old_psd = v_noise_psd

            # increment frame counter
            self._num_frames_processed += 1

            # return v_noisy_per
            return v_noisy_per, np.zeros(v_noisy_per.shape)

        else:
            # compute inverse GLR
            v_inv_glr = self._inv_glr_factor * \
                np.exp(-v_noisy_per / (self._v_old_psd + 1e-16) * self._inv_glr_exp_factor)

            # compute SPP (corresponds to line 2 in Algorithm 1, [1])
            v_spp = 1. / (1. + v_inv_glr)

            # stuck protection (corresponds to line 3 and 4 in Algorithm 1,
            # [1])
            self._v_smooth_prob = (1 - self._prob_smooth) * v_spp + \
                self._prob_smooth * self._v_smooth_prob
            v_mask = self._v_smooth_prob > 0.99
            v_spp[v_mask] = np.minimum(v_spp[v_mask], 0.99)

            # estimate noise periodogram (corresponds to line 5 in Algorithm
            # 1, [1])
            v_noise_per = (1. - v_spp) * v_noisy_per + \
                v_spp * self._v_old_psd
            # corresponds to line 6 in Algorithm 1, [1]
            v_noise_psd = (1. - self._fixed_smooth) * v_noise_per + \
                self._fixed_smooth * self._v_old_psd

        # update old noise PSD estimate
        self._v_old_psd = v_noise_psd

        # return v_noise_psd
        return v_noise_psd, v_spp

    def reset(self):
        """ Resets the internal states of the algorithm.

        """
        self._v_old_psd = np.zeros(self._frame_length // 2 + 1)
        self._v_smooth_prob = np.zeros(self._frame_length // 2 + 1)
        self._num_frames_processed = 0

    def from_stft(self, mat_per):
        """ Estimate the noise PSD from a matrix, i.e., a spectrogram, of noisy
        periodograms.

        :mat_per: noisy periodogram (|Y|^2, frames x coefficients)
        :returns: noise PSD estimate (\sigma^2_n, frames x coefficients)

        """
        # pre-allocate noise PSD
        mat_psd = np.zeros(mat_per.shape)

        for frame, per in enumerate(mat_per):
            mat_psd[frame] = self.update(per)

        self.reset()

        return mat_psd




def s2f(t_in_seconds, fs, hop_length):
    return int(t_in_seconds*fs/hop_length)

class VADEstimator():

    def __init__(self,
        frame_length=638,
        threshold_spp=.8,
        fs=32000,
        hop_length=160,
        lp_filter_freq_hz=4000,
        vad_buffer_size_s=.1,
        lookahead_s=.02
        ):

        self.threshold_spp = threshold_spp
        self.noise_estimator = SPPNoiseEstimator(frame_length=frame_length)
        self.lp_filter_freq_bin = int(lp_filter_freq_hz / fs * (frame_length//2+1))
        self.vad_buffer_size = s2f(vad_buffer_size_s, fs=fs, hop_length=hop_length)
        self.lookahead_s = lookahead_s
        self.hop_length = hop_length
        self.fs = fs

    def process(self, y):
        """
        y: F, T
        """
        y = y.cpu().numpy()
        vad_index = False
        spp = np.zeros(y.shape)
        vad_buffer = [False]*self.vad_buffer_size

        self.noise_estimator.reset()

        # while t < y.shape[-1] and not vad_index:
        for t in range(y.shape[-1]):
            _, spp[..., t] = self.noise_estimator.update(
                v_noisy_per=np.abs(y[..., t]**2)
            )
            vad = np.mean(spp[: self.lp_filter_freq_bin, t]) > self.threshold_spp and not vad_index
            vad_buffer = vad_buffer[1: ] + [vad]
            if np.array(vad_buffer).all():
                vad_index = t - s2f(self.lookahead_s, fs=self.fs, hop_length=self.hop_length)

        return spp, vad_index





if __name__ == "__main__":

    path = "/data/lemercier/databases/SIG-Challenge/blind_data"
    test_files = glob.glob(join(path, "*.wav"))[: 25]
    results_path = "/export/home/lemercier/code/sig-challenge-2023/.tests/vad"

    parser = ArgumentParser()
    # parser.add_argument("--tag", required=True)
    parser.add_argument("--nfft", default=640)
    parser.add_argument("--sr", default=32000)
    args = parser.parse_args()

    istft_kwargs = {
        "n_fft": args.nfft,
        "hop_length": args.nfft//4,
        "window": torch.hann_window(args.nfft),
        "center": True
    }
    stft_kwargs = {
        **istft_kwargs,
        "return_complex": True
    }

    os.makedirs(results_path, exist_ok=True)

    # Loop on files
    for sample in tqdm.tqdm(test_files):

        estimator = VADEstimator(frame_length=args.nfft, threshold_spp=0.8)

        y, sr = torchaudio.load(sample)
        if sr != args.sr:
            y = torch.from_numpy(librosa.resample(y.numpy(), orig_sr=sr, target_sr=SR))

        Y = torch.stft(y, **stft_kwargs).squeeze()
        spp, vad = estimator.process(Y)
        # print(vad)

        fig, axes = plt.subplots(3, 1)
        ax = axes.flatten()[0]
        ax.imshow(20*np.log10(Y.abs() + 1e-10), origin="lower")
        
        ax = axes.flatten()[1]
        ax.imshow(20*np.log10(spp + 1e-10), origin="lower")
        ax.axvline(x=vad, color="r")

        ax = axes.flatten()[2]
        ax.plot(np.mean(spp, axis=0), color="b")
        ax.plot(np.mean(spp[: 40, :], axis=0), color="k")
        ax.axvline(x=vad, color="r")
        ax.set_xlim([0, Y.shape[-1]])

        plt.savefig(join(results_path, os.path.basename(sample)[: -4] + ".png"))

        # soundfile.write(f'{os.path.dirname(sample)}/{os.path.basename(sample)[: -4]}' + f'_{args.tag}.wav', s.type(torch.float32).squeeze().unsqueeze(-1).numpy(), sr)
        # torchaudio.save(f'{os.path.dirname(sample)}/{os.path.basename(sample)[: -4]}' + f'_{args.tag}.wav', s.type(torch.int16).squeeze().unsqueeze(0), sr)
