"""
Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

Taken and adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sde_lib.py
"""
import abc
import warnings
import torch.nn.functional as F
import scipy.special as sc
import numpy as np
import torch

from sgmse.util.registry import Registry


SDERegistry = Registry("SDE")


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.

        Args:
            N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t, *args):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t, *args):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x|args)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape, *args):
        """Generate one sample from the prior distribution, $p_T(x|args)$ with shape `shape`."""
        pass


    @staticmethod
    @abc.abstractmethod
    def add_argparse_args(parent_parser):
        """
        Add the necessary arguments for instantiation of this SDE class to an argparse ArgumentParser.
        """
        pass

    def discretize(self, x, t, y, stepsize):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
            f, G
        """
        dt = stepsize
        #dt = 1 /self.N
        drift, diffusion = self.sde(x, t, y)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        
        
        return f, G

    def reverse(oself, score_model, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
            score_model: A function that takes x, t and y and returns the score.
            probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = oself.N
        T = oself.T
        sde_fn = oself.sde
        discretize_fn = oself.discretize

        # Build the class for reverse-time SDE.
        class RSDE(oself.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t, *args):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                rsde_parts = self.rsde_parts(x, t, *args)
                total_drift, diffusion = rsde_parts["total_drift"], rsde_parts["diffusion"]
                return total_drift, diffusion

            def rsde_parts(self, x, t, *args):
                sde_drift, sde_diffusion = sde_fn(x, t, *args)
                score = score_model(x, t, *args)
                score_drift = -sde_diffusion[:, None, None, None]**2 * score * (0.5 if self.probability_flow else 1.)
                diffusion = torch.zeros_like(sde_diffusion) if self.probability_flow else sde_diffusion
                total_drift = sde_drift + score_drift
                return {
                    'total_drift': total_drift, 'diffusion': diffusion, 'sde_drift': sde_drift,
                    'sde_diffusion': sde_diffusion, 'score_drift': score_drift, 'score': score,
                }

            def discretize(self, x, t, y, stepsize, scale_divide):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, t, y, stepsize)
                if torch.is_complex(G):
                    G = G.imag

                rev_f = f - G[:, None, None, None] ** 2 * score_model(x, t, y, scale_divide) * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()

    @abc.abstractmethod
    def copy(self):
        pass


@SDERegistry.register("ouve")
#From "Speech Enhancement and Dereverberation with Diffusion-based Generative Models"
class OUVESDE(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--sde-n", type=int, default=1000, help="The number of timesteps in the SDE discretization. 30 by default")
        parser.add_argument("--theta", type=float, default=1.5, help="The constant theta of the Ornstein-Uhlenbeck process. 1.5 by default.")
        parser.add_argument("--sigma-min", type=float, default=0.05, help="The minimum sigma to use. 0.05 by default.")
        parser.add_argument("--sigma-max", type=float, default=0.5, help="The maximum sigma to use. 0.5 by default.")
        return parser

    def __init__(self, theta, sigma_min, sigma_max, N=1000, **ignored_kwargs):
        """Construct an Ornstein-Uhlenbeck Variance Exploding SDE.

        Note that the "steady-state mean" `y` is not provided at construction, but must rather be given as an argument
        to the methods which require it (e.g., `sde` or `marginal_prob`).

        dx = -theta (y-x) dt + sigma(t) dw

        with

        sigma(t) = sigma_min (sigma_max/sigma_min)^t * sqrt(2 log(sigma_max/sigma_min))

        Args:
            theta: theta parameter.
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps
        """
        super().__init__(N)
        self.theta = theta
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.logsig = np.log(self.sigma_max / self.sigma_min)
        self.N = N
        self._T = 1

    def copy(self):
        return OUVESDE(self.theta, self.sigma_min, self.sigma_max, N=self.N)

    @property
    def T(self):
        return self._T

    def sde(self, x, t, y):
        drift = self.theta * (y - x)
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        diffusion = sigma * np.sqrt(2 * self.logsig)
        return drift, diffusion

    def _mean(self, x0, t, y):
        theta = self.theta
        exp_interp = torch.exp(-theta * t)[:, None, None, None]
        return exp_interp * x0 + (1 - exp_interp) * y

    def _std(self, t):
        # This is a full solution to the ODE for P(t) in our derivations, after choosing g(s) as in self.sde()
        sigma_min, theta, logsig = self.sigma_min, self.theta, self.logsig
        # could maybe replace the two torch.exp(... * t) terms here by cached values **t
        return torch.sqrt(
            (
                sigma_min**2
                * torch.exp(-2 * theta * t)
                * (torch.exp(2 * (theta + logsig) * t) - 1)
                * logsig
            )
            /
            (theta + logsig)
        )

    def marginal_prob(self, x0, t, y):
        return self._mean(x0, t, y), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self._std(torch.ones((y.shape[0],), device=y.device))
        z = torch.randn_like(y)
        #z = torch.zeros_like(y)
        x_T = y + z * std[:, None, None, None]
        return x_T, z

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for OU SDE not yet implemented!")

    def theta(self, t):
        return self.theta


    def diff_buffer_matrices(self, diff_buffer_length, x0, y, t_eps, fixed_buffer_steps):
        num_frames = x0.shape[-1]
        #get random time points as torch tensor: diff_buffer_length many 
        if not fixed_buffer_steps:
            diff_times =torch.rand(diff_buffer_length, device=x0.device) * (self.T - t_eps) + t_eps
            #sort the time points
            diff_times, _ = torch.sort(diff_times, dim=0)
        else:
            diff_times = torch.linspace(t_eps, self.T, int(diff_buffer_length.cpu().numpy()), device=x0.device)

        mean_interpolation = torch.zeros_like(x0.real)
        std =  self.get_diff_std(num_frames, diff_times).to(device=x0.device)

        # Compute stop indices for each batch element


        jj = len(diff_times) - 1
        for i in reversed(range(num_frames)):
            t = diff_times[jj]
            # Create mask for elements where i >= stop_indices
            # Update mean_interpolation and std where mask is True
            mean_interpolation[:, :, :, i] = 1-torch.exp(-self.theta*t)
            jj = jj - 1
            if jj == -1:
                break
        
        z = torch.randn_like(x0)
        z[:,:,:,:-diff_buffer_length] = 0


        pert_data = x0 * (1-mean_interpolation) + mean_interpolation*y + z * std
        
        #time embeddings [BS, diff buffer length]
        diff_times = F.pad(diff_times, (num_frames-diff_buffer_length, 0), mode='constant', value=0)
        diff_time_out = diff_times[None, :]
        
        std_out = std
        
        return pert_data, std_out, z, diff_time_out
        
    def get_meanevo_factor(self, difftimes):
        return 1 - torch.exp(-difftimes*self.theta)


    def get_diff_std(self, num_frames, diff_times):
        

        std = torch.zeros(1, num_frames)
        jj = len(diff_times) - 1
        for i in reversed(range(num_frames)):
            t = diff_times[jj]
            std[0, i] = self._std(t)
            jj = jj - 1
            if jj == -1:
                break

        out = std.unsqueeze(0).unsqueeze(0)
        
        return out





@SDERegistry.register("bbed")
#From "Reducing the Prior Mismatch of Stochastic Differential Equations for Diffusion-based Speech Enhancement"
class BBED(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--sde-n", type=int, default=30, help="The number of timesteps in the SDE discretization. 30 by default")
        parser.add_argument("--T_sampling", type=float, default=0.999, help="The T so that t < T during sampling in the train step.")
        parser.add_argument("--k", type=float, default = 2.6, help="base factor for diffusion term") 
        parser.add_argument("--theta", type=float, default = 0.52, help="squared scale factor for diffusion term")
        return parser

    def __init__(self, T_sampling, k, theta, N=1000, **kwargs):
        """Construct an Brownian Bridge SDE
        dx = (y-x)/(T-t) dt + dw
        Args:

            N: number of discretization steps
        """
        super().__init__(N)
        self.k = k
        self.logk = np.log(self.k)
        self.theta = theta
        self.N = N
        self.Eilog = sc.expi(-2*self.logk)
        self.T = T_sampling #for sampling in train step and inference
        self.Tc = 1 #for constructing the SDE, dont change this


    def copy(self):
        return BBED(self.T, self.k, self.theta, N=self.N)


    def T(self):
        return self.T
    
    def Tc(self):
        return self.Tc


    def sde(self, x, t, y):
        drift = (y - x)/(self.Tc - t)
        sigma = (self.k) ** t
        diffusion = sigma * np.sqrt(self.theta)
        return drift, diffusion


    def _mean(self, x0, t, y):
        time = (t/self.Tc)[:, None, None, None]
        mean = x0*(1-time) + y*time
        return mean

    def _std(self, t):
        t_np = t.cpu().detach().numpy()
        Eis = sc.expi(2*(t_np-1)*self.logk) - self.Eilog
        h = 2*self.k**2*self.logk
        var = (self.k**(2*t_np)-1+t_np) + h*(1-t_np)*Eis
        var = torch.tensor(var).to(device=t.device)*(1-t)*self.theta
        return torch.sqrt(var)

    def marginal_prob(self, x0, t, y):
        return self._mean(x0, t, y), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self._std(self.T*torch.ones((y.shape[0],), device=y.device))
        z = torch.randn_like(y)
        x_T = y + z * std[:, None, None, None]
        return x_T, z


    def theta(self, t):
        return 1/(1-t)
    
    def diff_buffer_matrices(self, diff_buffer_length, x0, y, t_eps, fixed_buffer_steps):
        num_frames = x0.shape[-1]
        #get random time points as torch tensor: diff_buffer_length many 
        if not fixed_buffer_steps:
            diff_times =torch.rand(diff_buffer_length, device=x0.device) * (self.T - t_eps) + t_eps
            #sort the time points
            diff_times, _ = torch.sort(diff_times, dim=0)
        else:
            diff_times = torch.linspace(t_eps, self.T, int(diff_buffer_length.cpu().numpy()), device=x0.device)

        mean_interpolation = torch.zeros_like(x0.real)
        std = self.get_diff_std(num_frames, diff_times).to(device=x0.device)

        # Compute stop indices for each batch element


        jj = len(diff_times) - 1
        for i in reversed(range(num_frames)):
            t = diff_times[jj]
            # Create mask for elements where i >= stop_indices
            # Update mean_interpolation and std where mask is True
            mean_interpolation[:, :, :, i] = t / self.T
            jj = jj - 1
            if jj == -1:
                break
        
        z = torch.randn_like(x0)

        #masking z: take only values where std != 0
        z[:,:,:,:int(-diff_buffer_length)] = 0
        pert_data = x0 * (1-mean_interpolation) + mean_interpolation*y + z * std
        
        #time embeddings [BS, diff buffer length]
        diff_times = F.pad(diff_times, (num_frames-int(diff_buffer_length), 0), mode='constant', value=0)
        return pert_data, std, z, diff_times[None,:]

    def get_diff_std(self, num_frames, diff_times):
        std = torch.zeros(1, num_frames)
        jj = len(diff_times) - 1
        for i in reversed(range(num_frames)):
            t = diff_times[jj]
            std[0, i] = self._std(t)
            jj = jj - 1
            if jj == -1:
                break
        out = std.unsqueeze(0).unsqueeze(0)
        
        return out

    def get_meanevo_factor(self, difftimes):
        return difftimes
