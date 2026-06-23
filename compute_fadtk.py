import torch
import numpy as np
from tqdm import tqdm
import soundfile
from numpy.lib.scimath import sqrt as scisqrt
from scipy import linalg
import glob
from fadtk.fad import FrechetAudioDistance
from fadtk.model_loader import CLAPLaionModel
import torchaudio


def calc_frechet_distance(mu1, cov1, mu2, cov2, eps=1e-6):
    """
    Adapted from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    
    Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
            inception net (like returned by the function 'get_predictions')
            for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
            representative data set.
    -- cov1: The covariance matrix over activations for generated samples.
    -- cov2: The covariance matrix over activations, precalculated on an
            representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    cov1 = np.atleast_2d(cov1)
    cov2 = np.atleast_2d(cov2)

    assert mu1.shape == mu2.shape, \
        f'Training and test mean vectors have different lengths ({mu1.shape} vs {mu2.shape})'
    assert cov1.shape == cov2.shape, \
        f'Training and test covariances have different dimensions ({cov1.shape} vs {cov2.shape})'

    diff = mu1 - mu2

    # Product might be almost singular
    # NOTE: issues with sqrtm for newer scipy versions
    # using eigenvalue method as workaround
    covmean_sqrtm, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)
    
    # eigenvalue method
    D, V = linalg.eig(cov1.dot(cov2))
    covmean = (V * scisqrt(D)) @ linalg.inv(V)

    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
            'adding %s to diagonal of cov estimates') % eps
        offset = np.eye(cov1.shape[0]) * eps
        covmean = linalg.sqrtm((cov1 + offset).dot(cov2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    tr_covmean_sqrtm = np.trace(covmean_sqrtm)
    if np.iscomplexobj(tr_covmean_sqrtm):
        if np.abs(tr_covmean_sqrtm.imag) < 1e-3:
            tr_covmean_sqrtm = tr_covmean_sqrtm.real

    if not(np.iscomplexobj(tr_covmean_sqrtm)):
        delt = np.abs(tr_covmean - tr_covmean_sqrtm)
        if delt > 1e-3:
            pass

    return (diff.dot(diff) + np.trace(cov1)
            + np.trace(cov2) - 2 * tr_covmean)



def fadtk_score_from_arrays(fadtk_model, clean_audio, enhanced_audio, sr=16000):
    """Compute FAD directly from numpy arrays"""
    # Get embeddings for clean audio
    clean_emb = fadtk_model.ml._get_embedding(clean_audio)
    
    # Get embeddings for enhanced audio
    enhanced_emb = fadtk_model.ml._get_embedding(enhanced_audio)
    
    # Calculate FAD
    mu1, sigma1 = torch.mean(clean_emb), torch.cov(clean_emb)
    mu2, sigma2 = torch.mean(enhanced_emb), torch.cov(enhanced_emb)
    
    mu1 = mu1.detach().cpu().numpy()
    mu2 = mu2.detach().cpu().numpy()
    sigma1 = sigma1.detach().cpu().numpy()
    sigma2 = sigma2.detach().cpu().numpy()
    
    return calc_frechet_distance(mu1, sigma1, mu2, sigma2)


class FADTK_embedding_cacher():
    def __init__(self, delays, clean_test_dir, diff_gate_length):
        self.fadtk_model = FrechetAudioDistance(
                CLAPLaionModel('audio'),  # or another model
                audio_load_worker=1,
                load_model=True
            )
        self.num_test_files = len(clean_test_dir)
        self.delays = delays
        self.cached_embeddings = {f'{x}': [] for x in delays}
        self.cached_embeddings[f'{diff_gate_length}'] = []
        self.clean_test_dir = clean_test_dir
        self.embdim =512
        self.compute_emb_test_dir()
    
    
    def compute_embd_delay(self, delay, x_hat):
        cache_dict = self.cached_embeddings[delay]
        x = torchaudio.functional.resample(x_hat, 16000, 48_000)
        x = x.squeeze().detach().cpu().numpy()
        emb = self.fadtk_model.ml._get_embedding(x)
        cache_dict.append(emb.detach().cpu().numpy())
        
    
    def compute_fad_value(self, kk):
        #compute mu and sigma 
        cached_list = self.cached_embeddings[kk]
        all_clean_embeddings = np.concatenate(cached_list, axis=0)
        mu, sigma = np.mean(all_clean_embeddings, axis=0), np.cov(all_clean_embeddings, rowvar=False)
        fad_value = calc_frechet_distance(self.mu_clean, self.sigma_clean, mu, sigma)
        return fad_value
        
        

    def compute_emb_test_dir(self):
        #load test dir and compute embeddings
        clean_emb_cached =[]
        
        clean_files  = sorted(glob.glob('{}/**/*.wav'.format(self.clean_test_dir)), key=lambda x: x.split('/')[-1])
        
        if len(clean_files)==0:
            clean_files  = sorted(glob.glob('{}/*.wav'.format(self.clean_test_dir)), key=lambda x: x.split('/')[-1])
        for _, clean_file in tqdm(enumerate(clean_files)):
            # Load wav
            x, fs = torchaudio.load(clean_file)
            assert fs == 16_000
            #resample
            x = torchaudio.functional.resample(x, fs, 48_000) #for CLAP must be 48000
            #torchaudio.save('test.wav', x.cpu(), 48000)
            x = x.squeeze().detach().cpu().numpy()
            
            emb = self.fadtk_model.ml._get_embedding(x)
            clean_emb_cached.append(emb.detach().cpu().numpy())
            if _ == 1:
                break


        #compute mu and sigma
        all_clean_embeddings = np.concatenate(clean_emb_cached, axis=0)
        self.mu_clean, self.sigma_clean = np.mean(all_clean_embeddings, axis=0), np.cov(all_clean_embeddings, rowvar=False)
        
        
        