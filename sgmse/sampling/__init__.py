# Adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sampling.py
"""Various sampling methods."""
from scipy import integrate
import torch

from .predictors import Predictor, PredictorRegistry, ReverseDiffusionPredictor
from .correctors import Corrector, CorrectorRegistry
import numpy as np
import matplotlib.pyplot as plt


__all__ = [
    'PredictorRegistry', 'CorrectorRegistry', 'Predictor', 'Corrector',
    'get_sampler'
]


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def get_pc_sampler(
    predictor_name, corrector_name, sde, score_fn, Y, Y_prior=None,
    denoise=True, eps=3e-2, snr=0.1, corrector_steps=1, probability_flow: bool = False,
    intermediate=False, timestep_type=None, output_scale='time', **kwargs
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
        predictor_name: The name of a registered `sampling.Predictor`.
        corrector_name: The name of a registered `sampling.Corrector`.
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        denoise: If `True`, add one-step denoising to the final samples.
        eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
        snr: The SNR to use for the corrector. 0.1 by default, and ignored for `NoneCorrector`.
        N: The number of reverse sampling steps. If `None`, uses the SDE's `N` property by default.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    predictor_cls = PredictorRegistry.get_by_name(predictor_name)
    corrector_cls = CorrectorRegistry.get_by_name(corrector_name)
    predictor = predictor_cls(sde, score_fn, probability_flow=probability_flow)
    corrector = corrector_cls(sde, score_fn, snr=snr, n_steps=corrector_steps)

    def pc_sampler(Y_prior=Y_prior, timestep_type=timestep_type, output_scale=output_scale):
        """The PC sampler function."""
        with torch.no_grad():
            
            if Y_prior == None:
                Y_prior = Y
            
            xt, _ = sde.prior_sampling(Y_prior.shape, Y_prior)
            timesteps = timesteps_space(sde.T, sde.N,eps, Y.device, type=timestep_type)
            xt = xt.to(Y_prior.device)
            
            
            for i in range(len(timesteps)):
                t = timesteps[i]
                if i != len(timesteps) - 1:
                    stepsize = t - timesteps[i+1]
                else:
                    stepsize = timesteps[-1]

                vec_t = torch.ones(Y.shape[0], device=Y.device) * t

                xt, xt_mean = corrector.update_fn(xt, vec_t, Y, output_scale)
                xt, xt_mean = predictor.update_fn(xt, vec_t, Y, stepsize, output_scale)
            x_result = xt_mean if denoise else xt
            ns = len(timesteps) * (corrector.n_steps + 1)
            return x_result, ns
    
    if intermediate:
        return pc_sampler_intermediate
    else:
        return pc_sampler



def timesteps_space(sdeT, sdeN,  eps, device, type='linear'):

    timesteps = torch.linspace(sdeT, eps, sdeN, device=device)
    if type == 'linear':
        return timesteps
    elif type == 'squared':
        timesteps = torch.square(timesteps)
    elif type == 'log':
        #-1.5 = theta hardcoded
        timesteps = -1/1.5*torch.log(1-timesteps)
        stopindex = -1
        startindex = -1
        for i in range(len(timesteps)):
            if timesteps[i] <= 1 and startindex == -1:
                startindex = i
            if timesteps[i] < eps and stopindex== -1:
                stopindex=i
        timesteps = timesteps[startindex:stopindex]
    elif type == 'hc1':
        #hardcoded for ousvp
        aux = torch.tensor([eps, 2*eps, 3*eps, 4*eps, 5*eps, 6*eps, 7*eps, 8*eps, 9*eps, 10*eps,
                      11*eps, 12*eps, 13*eps, 14*eps, 15*eps, 16*eps, 17*eps, 18*eps, 19*eps, 20*eps,
                      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
        timesteps = aux.to(timesteps.device)
        timesteps = reversed(timesteps)
    
    return timesteps







def get_ode_sampler(
    sde, score_fn, y, Y_prior=None, inverse_scaler=None, 
    denoise=True, rtol=1e-5, atol=1e-5, timestep_type = None,
    method='RK45', eps=3e-2, device='cuda', **kwargs
):
    """Probability flow ODE sampler with the black-box ODE solver.

    Args:
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        inverse_scaler: The inverse data normalizer.
        denoise: If `True`, add one-step denoising to final samples.
        rtol: A `float` number. The relative tolerance level of the ODE solver.
        atol: A `float` number. The absolute tolerance level of the ODE solver.
        method: A `str`. The algorithm used for the black-box ODE solver.
            See the documentation of `scipy.integrate.solve_ivp`.
        eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
        device: PyTorch device.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    predictor = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    rsde = sde.reverse(score_fn, probability_flow=True)

    def denoise_update_fn(x):
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor.update_fn(x, vec_eps, y, 0.03)
        return x

    def drift_fn(x, t, y):
        """Get the drift function of the reverse-time SDE."""
        return rsde.sde(x, t, y)[0]

    def ode_sampler(z=None, Y_prior=Y_prior, **kwargs):
        """The probability flow ODE sampler with black-box ODE solver.

        Args:
            model: A score model.
            z: If present, generate samples from latent code `z`.
        Returns:
            samples, number of function evaluations.
        """
        with torch.no_grad():
            # If not represent, sample the latent code from the prior distibution of the SDE.


            if Y_prior == None:
                Y_prior = y
            
            xt, _ = sde.prior_sampling(Y_prior.shape, Y_prior)
            x = xt.to(Y_prior.device)

            def ode_func(t, x):
                x = from_flattened_numpy(x, y.shape).to(device).type(torch.complex64)
                vec_t = torch.ones(y.shape[0], device=x.device) * t
                drift = drift_fn(x, vec_t, y)
                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(
                ode_func, (sde.T, eps), to_flattened_numpy(x),
                rtol=rtol, atol=atol, method=method, **kwargs
            )
            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape(y.shape).to(device).type(torch.complex64)

            # Denoising is equivalent to running one predictor step without adding noise
            if denoise:
                x = denoise_update_fn(x)

            if inverse_scaler is not None:
                x = inverse_scaler(x)
            return x, nfe

    return ode_sampler
