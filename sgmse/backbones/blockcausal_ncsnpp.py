"""
TRAINING checkpoints with center-shift to base inference experiments on
"""
import einops
import torch.nn as nn
import functools
import torch
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
from math import ceil
from .ncsnpp_utils import up_or_down_sampling
from .ncsnpp_utils.utils import variance_scaling
import math
from .shared import BackboneRegistry




#def positionalencoding1d(in_tensor, gls):
    #"""
    #:param d_model: dimension of the model
    #:param length: length of positions
    #:return: length*d_model position matrix
    #"""
    #x = in_tensor[0,0,:,:]
    #d_model = x.shape[-2]
    #length = x.shape[-1]
    #if d_model % 2 != 0:
    #    raise ValueError("Cannot use sin/cos positional encoding with "
    #                     "odd dim (got dim={:d})".format(d_model))
    #pe = torch.zeros_like(x)
    #position = torch.arange(0, length).unsqueeze(1)
    #div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
    #                     -(math.log(10000.0) / d_model)))
    #pe[0::2, :] = torch.sin(position.float() * div_term)
    #pe[1::2, :] = torch.cos(position.float() * div_term)
    
    #TODO expand pe so that last dimension is now of shape in_tensor.shape[-1] by repeating it.

    #return pe.unsqueeze(0).unsqueeze(0)
    
    
def positionalencoding1d(d_model, length, input_tensor):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    
    pe = pe.type(input_tensor.type())
    pe = pe.to(device=input_tensor.device)
    pe =  torch.transpose(pe, 1, 0)
    repeats=  math.ceil(input_tensor.shape[-1] / length)
    pe = pe.repeat(1, repeats)
    diff = length*repeats - input_tensor.shape[-1]
    if diff > 0:
        pe = pe[..., diff:]
    return pe.unsqueeze(0).unsqueeze(0)



def init_weights(instance, scale=1.0):
    scale = 1e-10 if scale == 0 else scale
    # varicane_scaling ported from JAX: jax.nn.initializers.variance_scaling
    # initializer that adapts its scale to the shape of the weights tensor.
    # same as torch.nn.init.kaiming_uniform_ with edit fan_avg mode
    default_init = variance_scaling(scale, "fan_avg", "uniform")
    instance.weight.data = default_init(instance.weight.data.shape)
    nn.init.zeros_(instance.bias)


class Linear_init(nn.Module):
    """Linear layer with default initialization (kaiming_uniform with
    fan_avg mode / bias zeros)."""

    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype=None, scale=1.0
    ):
        super().__init__()

        self.lin_layer = nn.Linear(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )

        init_weights(self.lin_layer, scale=scale)

    def forward(self, x):
        return self.lin_layer(x)


class Conv2d_realtime(nn.Conv2d):
    """
    real-time implementation of 2D convolution

    Args:
        in_channels: int
            number of input channels
        out_channels: int
            number of output channels
        kernel_size: int | tuple[int]
            size of convolution kernel.
        stride: int | tuple[int]
            stride of convolution.
        dilation: int | tuple[int]
            dilation of convolution kernel.
        v_pad_mode: str | tuple[int]
            str options: ["default", "full"]
            tuple[int]: (top_pad, bottom_pad)
            choice of vertical padding mode
        causal: bool (default: True)
            True -> causal convolution mode,
            False -> regular convolution mode.
        dtype: dtype
            data type of inputs, buffers and weights etc.
        n_buffs: int
            number of (input-)buffers to maintain.

    Methods:
        forward(input):
            processes input

        reset():
            resets all buffers and the current in/out indices
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        kernel_size,
        stride=1,
        dilation=1,
        v_pad_mode="default",
        causal=True,
        dtype=None,
        n_buffs=1,
        **kwargs,
    ):

        self.n_in_buffs = n_buffs
        self.dtype = dtype
        self.causal = causal
        ker_height = kernel_size if type(kernel_size) == int else kernel_size[0]
        ker_width = kernel_size if type(kernel_size) == int else kernel_size[1]
        dil_height = dilation if type(dilation) == int else dilation[0]
        dil_width = dilation if type(dilation) == int else dilation[1]
        self.ker_w_dil = (ker_width - 1) * dil_width + 1
        self.ker_h_dil = (ker_height - 1) * dil_height + 1

        if type(v_pad_mode) == str:
            assert v_pad_mode in [
                "default",
                "full",
            ], f"unknown padding mode {v_pad_mode}, choose from ['default', 'full']"
            self.top_pad, self.bottom_pad = None, None
            self.v_pad_mode = v_pad_mode
        else:
            assert len(v_pad_mode) == 2 and all(
                isinstance(p, int) for p in v_pad_mode
            ), "invalid choice for v_padding"
            self.top_pad, self.bottom_pad = v_pad_mode
            self.v_pad_mode = "manual"

        self.layer_stride_h = stride if type(stride) == int else stride[0]
        self.layer_stride = stride if type(stride) == int else stride[1]

        assert (
            self.ker_w_dil >= self.layer_stride
        ), f"expected 'dilated kernel size' to be be greater than 'stride' but got {self.ker_w_dil} < {self.layer_stride}"

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            dtype=self.dtype,
            **kwargs,
        )

        ### preparing list of input buffers (one for each in stream)
        self.internal_buff = None
        self.len_internal_buff = self.ker_w_dil - 1

        ### index to keep track of currently relevant input buffer
        self.curr_in_idx = -1

        self.do_padding = True
        if kernel_size == 1:
            self.do_padding = False

    def reset(self):
        """
        resets all indices and buffers
        """
        self.curr_in_idx = -1
        self.internal_buff = None

    def forward(self, input, *, train=True) -> torch.tensor:
        """
        Args:
            input: tensor
                new data

        Returns:
            tensor: next convolution output samples considering most recent input sample
            tuple[int]: top padding and bottom padding applied in conv
        """
        ### determine top and bottom padding, if not specified explicitely
        if self.v_pad_mode == "manual":  ### NOTE reliability completely up to user
            top_pad, bottom_pad = self.top_pad, self.bottom_pad

        elif self.v_pad_mode == "default":
            top_pad = (self.ker_h_dil - 1) // 2
            bottom_pad = (self.ker_h_dil) // 2
            rem = (input.shape[-2] - 1) % self.layer_stride_h
            if rem:
                bottom_pad += self.layer_stride_h - rem

        elif self.v_pad_mode == "full":  ### maximum padding maintaining overlap
            top_pad = self.ker_h_dil - 1
            max_steps = ceil((input.shape[-2] + top_pad) / self.layer_stride_h)
            bottom_pad = self.layer_stride_h * max_steps - input.shape[-2] + 1
        else:
            raise Exception("unknown padding mode")

        vert_pad = (0, 0, top_pad, bottom_pad)

        if self.do_padding:
            if self.causal:
                ### determine left padding
                max_steps = ceil(input.shape[-1] / self.layer_stride)
                wanted_length = (max_steps - 1) * self.layer_stride + self.ker_w_dil
                left_pad = wanted_length - input.shape[-1]

                ### pad input data
                input = F.pad(input, pad=(left_pad, 0), mode="constant", value=0)  # time
            else:
                input = F.pad(input, pad=(1, 1), mode="constant", value=0)

            input = F.pad(input, pad=vert_pad, mode="replicate")  # frequency

        return super().forward(input), (top_pad, bottom_pad)


class Conv2dTranspose_realtime(nn.ConvTranspose2d):
    """
    real-time implementation of transposed 2D convolution

    Args:
        in_channels: int
            number of input channels
        out_channels: int
            number of output channels
        kernel_size: list[int | tuple[int]]
            size of convolution kernel
        stride: int | tuple[int]
            stride of convolution.
        dilation: int | tuple[int]
            dilation of convolution kernel.
        causal: bool (default: True)
            True -> causal transposed convolution mode,
            False -> regular transposed convolution mode.
        dtype: dtype
            data type of inputs, buffers and weights etc.
        n_buffs: int
            number of (unfinished output-)buffers to maintain.
        total_stride: dtype
            product of all temporal strides up this layer
            (including current layer).

    Methods:
        forward(input):
            processes input

        reset():
            resets all indices and buffers
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        kernel_size,
        stride=1,
        dilation=1,
        causal=True,
        dtype=None,
        n_buffs=1,
        total_stride=1,
        **kwargs,
    ):

        self.dtype = dtype
        self.causal = causal
        self.n_buffs = n_buffs
        self.total_stride = total_stride

        kernel_height = kernel_size if type(kernel_size) == int else kernel_size[0]
        kernel_width = kernel_size if type(kernel_size) == int else kernel_size[1]
        dilation_height = dilation if type(dilation) == int else dilation[0]
        dilation_width = dilation if type(dilation) == int else dilation[1]
        self.ker_w_dil = (kernel_width - 1) * dilation_width + 1
        self.ker_h_dil = (kernel_height - 1) * dilation_height + 1

        self.layer_stride_freq = stride if type(stride) == int else stride[0]
        self.layer_stride_time = stride if type(stride) == int else stride[1]

        assert (
            self.ker_w_dil >= self.layer_stride_time
        ), f"expected 'dilated kernel size' to be be greater than 'stride' but got {self.ker_w_dil} <= {self.layer_stride_time}"

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            dtype=self.dtype,
            **kwargs,
        )

        self.inval_len = self.ker_w_dil - self.layer_stride_time

        ### create left padding lookup list
        self.prev_stride = total_stride // self.layer_stride_time

        ### based on the output buffer used, determine the amount of implicit left_padding
        # used in corresponding down-layer and add to look-up list
        self.left_pad = [
            self.layer_stride_time - 1 - (i // self.prev_stride)
            for i in range(self.total_stride)
        ]

        if self.bias != None:  # preventing dimension errors in broadcasting
            self.bias_reshaped = self.bias.reshape(-1, 1, 1).cuda()

    def reset(self):
        """
        resets all indices and buffers
        """
        self.curr_idx = -1
        self.internal_buff = None

    def forward(self, input, input_idx=None, *, train=True, v_pad=None) -> torch.tensor:
        """
        Args:
            input: tensor
                new data
            v_pad (optional): tuple[int]
                vertical padding (top, bottom) used in corresponding DownConv

        Returns:
            tensor: next convolution output samples considering newly arrived input
        """
        ### ensure same padding as in corresponding DownConv
        if v_pad != None:
            top_pad, bottom_pad = v_pad
        else:
            top_pad, bottom_pad = 0, 0

        if self.causal:
            if bottom_pad:
                output = super().forward(input)[..., top_pad:-bottom_pad, :]
            else:
                output = super().forward(input)[..., top_pad:, :]

            self.curr_idx = input_idx % self.total_stride
            left_excess = self.left_pad[self.curr_idx]
            right_excess = self.ker_w_dil - self.layer_stride_time
            if right_excess:
                output = output[..., left_excess:-right_excess]
            else:
                output = output[..., left_excess:]
            return output

        else:
            self.curr_idx = input_idx % self.total_stride
            left_excess = self.left_pad[self.curr_idx]
            right_excess = self.ker_w_dil - self.layer_stride_time
            # print(input.shape, input_idx, self.curr_idx, left_excess, right_excess, top_pad, bottom_pad)
            if right_excess == 2:
                left_excess = 1
                right_excess = 1
            
            output = super().forward(input)[..., top_pad:-bottom_pad, left_excess:-right_excess]  # remove left/right padding
            return output


class FastCumulativeGroupNorm(torch.nn.Module):
    def __init__(self, num_groups: int, num_channels: int, n_buffs: int, eps: float = 1e-5):
        super().__init__()
        self.eps, self.num_channels = eps, num_channels

        # adjust groups to divide channels
        if num_channels % num_groups != 0:
            for g in range(num_groups, num_channels + 1):
                if num_channels % g == 0:
                    num_groups = g
                    break
        self.num_groups = num_groups
        self.group_size = num_channels // num_groups

        self.gain = torch.nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = torch.nn.Parameter(torch.zeros(1, num_channels, 1, 1))


        # time‐counts and channel divisor as buffers
        counts = torch.arange(1, n_buffs+1, dtype=torch.float32).view(1,1,n_buffs)
        self.register_buffer('counts', counts)
        self.register_buffer('inv_div', torch.tensor(1.0 / self.group_size))

    def forward(self, x: torch.Tensor, train=True):
        # # ensure gain/bias live on the same device as x during compilation
        # gain = self.gain.to(x.device)
        # bias = self.bias.to(x.device)

        N, C, F, T = x.shape
        G, gs = self.num_groups, self.group_size
        xg = x.view(N, G, gs, F, T)

        sum_x  = xg.sum(dim=(2,3))
        sum_x2 = (xg * xg).sum(dim=(2,3))

        csum  = sum_x.cumsum(dim=-1)
        csum2 = sum_x2.cumsum(dim=-1)
        cnt = torch.arange(1, T+1, device=x.device, dtype=x.dtype).view(1,1,T)

        mu  = csum.mul(self.inv_div).div(cnt * F)
        var = csum2.mul(self.inv_div).div(cnt * F) - mu**2

        mu   = mu.view(N, G, 1, 1, T)
        var  = var.view(N, G, 1, 1, T)
        invs = torch.rsqrt(var + self.eps)

        xn = (xg - mu) * invs
        out = xn.view(N, C, F, T)
        return out * self.gain + self.bias
    
    def train(self, mode=True):
        # Turn backpropagation on (mode=True, training) or off (mode=False, evaluation)
        # for learnable parameters gain and bias
        self.gain.requires_grad_(mode)
        self.bias.requires_grad_(mode)


class TemporalEncoding(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, temb_dim=512, embed_dim=128, scale=16.0):  # embed_dim=256
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim) * scale, requires_grad=False)

        self.lin_layer1 = Linear_init(embed_dim * 2, temb_dim)
        self.lin_layer2 = Linear_init(temb_dim, temb_dim)
        self.swish = nn.SiLU()

    def forward(self, t):
        t_emb = torch.log(t + 1e-6)
        t_emb = t[:, :, None] * self.W[None, None, :] * 2 * np.pi
        t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
        t_emb = self.lin_layer1(t_emb)
        t_emb = self.swish(self.lin_layer2(t_emb))
        return t_emb


        

class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch=None,
        temb_dim=512,
        *,
        up=False,
        down=False,
        stride=(1, 1),
        transpose=False,
        is_last_transposed_conv=False,
        dropout=0.0,
        fir_kernel=(1, 3, 3, 1),
        fir=True,
        act=nn.SiLU(),
        causal=False,
        force_native=False,
        force_group_norm=False,
        force_cum_group_norm=False,
        no_group_norm=False,
        total_stride=1,
        global_stride=1,
        dilation_freq = 1,
        time_emb_reduction='conv',
        min_groups=32,
        pe_abs = None,
        pe_abs_ch_in = 128,
        ith_block = 0,
    ):
        super().__init__()

        self.causal = causal
        self.force_native = force_native
        self.force_group_norm = force_group_norm
        self.force_cum_group_norm = force_cum_group_norm
        self.no_group_norm = no_group_norm
        self.transpose = transpose
        self.is_last_transposed_conv = is_last_transposed_conv
        self.dilation  = (dilation_freq, 1)
        self.layer_stride_time = stride if type(stride) == int else stride[1]
        self.time_emb_reduction = time_emb_reduction

        # determining number of skips/down_buffs
        self.total_stride = total_stride
        self.n_in_buffs = total_stride // self.layer_stride_time

        # max number values required in skip
        self.lower_stride = global_stride // total_stride * self.layer_stride_time

        self.global_stride = global_stride  # for number of up_buffs

        if not self.transpose:
            ### preparing list of input buffers (one for each in stream)
            self.in_buffs = [None for _ in range(self.n_in_buffs)]
            ### index to keep track of currently relevant input/output buffer
            self.curr_in_idx = -1

        out_ch = out_ch if out_ch else in_ch
        if (self.causal and not self.force_group_norm) or self.force_cum_group_norm:
            self.GroupNorm_0 = FastCumulativeGroupNorm(
                num_groups=min(in_ch // 4, min_groups),
                num_channels=in_ch,
                eps=1e-6,
                n_buffs=self.global_stride if self.transpose else self.n_in_buffs,
            )
        elif self.no_group_norm:
            self.GroupNorm_0 = nn.Identity()
        else:
            self.GroupNorm_0 = nn.GroupNorm(
                num_groups=min(in_ch // 4, min_groups), num_channels=in_ch, eps=1e-6
            )
        self.up = up
        self.down = down
        self.fir_kernel = fir_kernel
        self.fir = fir

        if not self.transpose:
            self.Conv_0 = Conv2d_realtime(
                in_ch,
                out_ch,
                stride=1,
                kernel_size=3,
                dilation = self.dilation,
                causal=self.causal,
                n_buffs=self.n_in_buffs,
            )
        else:
            self.Conv_0 = Conv2dTranspose_realtime(
                in_ch,
                out_ch,
                stride=1,
                kernel_size=3,
                dilation = self.dilation,
                causal=self.causal,
                total_stride=self.total_stride,
                n_buffs=self.global_stride,
            )
            


        if temb_dim is not None:
            if self.down:
                self.stride_for_time_emb = self.total_stride//self.layer_stride_time
            elif self.up:
                self.stride_for_time_emb = self.total_stride
            else: #bottleneck
                self.stride_for_time_emb = self.total_stride//self.layer_stride_time
            if self.time_emb_reduction == 'conv':
                self.Conv_time = torch.nn.Conv1d(temb_dim, out_ch, kernel_size=3,  
                                                stride=self.stride_for_time_emb,  padding=1)
            elif self.time_emb_reduction == 'resample':
                self.Conv_time = torch.nn.Conv1d(temb_dim, out_ch, kernel_size=1,  
                                                stride=1,  padding='same')
            else:
                raise ValueError('Time emb reduction must be either resample or conv')
          
        #absolute encoding
        if pe_abs:
            if self.down:
                self.stride_for_peabs_emb = self.total_stride//self.layer_stride_time
                stride_for_peabs_freq = 2**(ith_block+1)
            elif self.up:
                self.stride_for_peabs_emb = self.total_stride
                stride_for_peabs_freq = 2**(ith_block-1)
            else: #bottleneck
                self.stride_for_peabs_emb = self.total_stride//self.layer_stride_time
            

            self.conv_pe_abs = torch.nn.Conv2d(pe_abs_ch_in, out_ch, kernel_size=3,  
                                            stride=(stride_for_peabs_freq, self.stride_for_peabs_emb),  
                                            padding=1)

        if not self.fir:
            if self.down:
                self.conv_freq1 = torch.nn.Conv2d(in_ch, in_ch, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), padding_mode='reflect')
                self.conv_freq2 = torch.nn.Conv2d(in_ch, in_ch, stride=(2, 1), kernel_size=(3, 1), padding=(1, 0), padding_mode='reflect')
            elif self.up:
                self.conv_freq1 = torch.nn.ConvTranspose2d(in_ch, in_ch, stride=(2, 1), kernel_size=(3, 1), padding=(1, 0), output_padding=(1,0))
                self.conv_freq2 = torch.nn.ConvTranspose2d(in_ch, in_ch, stride=(2, 1), kernel_size=(3, 1), padding=(1, 0), output_padding=(1,0))

        
        
        #self.Dense_0 = Linear_init(temb_dim, out_ch)
        if (self.causal and not self.force_group_norm) or self.force_cum_group_norm:
            self.GroupNorm_1 = FastCumulativeGroupNorm(
                num_groups=min(out_ch // 4, min_groups),
                num_channels=out_ch,
                eps=1e-6,
                n_buffs=self.global_stride if self.transpose else self.n_in_buffs,
            )
        elif self.no_group_norm:
            self.GroupNorm_1 = nn.Identity()
        else:
            self.GroupNorm_1 = nn.GroupNorm(
                num_groups=min(out_ch // 4, min_groups), num_channels=out_ch, eps=1e-6
            )
        self.Dropout_0 = nn.Dropout(dropout)

        if not self.transpose:
            self.Conv_1 = Conv2d_realtime(
                out_ch,
                out_ch,
                kernel_size=(3, max(3, stride[1]+1)),
                stride=stride,
                dilation = self.dilation,
                causal=self.causal,
                n_buffs=self.n_in_buffs,
            )
            self.Conv_2 = Conv2d_realtime(
                in_ch,
                out_ch,
                kernel_size=(3, max(3, stride[1]+1)),
                stride=stride,
                dilation = self.dilation,
                causal=self.causal,
                n_buffs=self.n_in_buffs,
            )
        else:
            self.Conv_1 = Conv2dTranspose_realtime(
                out_ch,
                out_ch,
                kernel_size=(3, max(3, stride[1]+1)),
                stride=stride,
                dilation = self.dilation,
                causal=self.causal,
                total_stride=self.total_stride,
                n_buffs=self.global_stride,
            )
            self.Conv_2 = Conv2dTranspose_realtime(
                in_ch,
                out_ch,
                kernel_size=(3, max(3, stride[1]+1)),
                stride=stride,
                dilation = self.dilation,
                causal=self.causal,
                total_stride=self.total_stride,
                n_buffs=self.global_stride,
            )

        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch

    def return_in_buffer(self):
        pass

    def reset(self):
        self.Conv_0.reset()
        self.Conv_1.reset()
        self.Conv_2.reset()
        if type(self.GroupNorm_0) == FastCumulativeGroupNorm:
            self.GroupNorm_0.reset()
        if type(self.GroupNorm_1) == FastCumulativeGroupNorm:
            self.GroupNorm_1.reset()

    def forward(self, x, temb=None, pe_abs=None, v_pad=None, input_idx=-1, train=True):

        if type(self.GroupNorm_0) == FastCumulativeGroupNorm:
            h = self.act(self.GroupNorm_0(x))
        else:
            h = self.act(self.GroupNorm_0(x))
        if self.up:
            if self.fir:
                h = up_or_down_sampling.upsample_1d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.upsample_1d(x, self.fir_kernel, factor=2)
            else:
                h = self.conv_freq1(h)
                x = self.conv_freq2(x)
        elif self.down:
            if self.fir:
                h = up_or_down_sampling.downsample_1d(h, self.fir_kernel, factor=2)
                x = up_or_down_sampling.downsample_1d(x, self.fir_kernel, factor=2)
            else:
                h = self.conv_freq1(h)
                x = self.conv_freq2(x)
        if not self.transpose:
            h, _ = self.Conv_0(h)
        else:
            h = self.Conv_0(h, v_pad=v_pad, input_idx=input_idx)

        # Add bias to each feature map conditioned on the time embedding
        #if temb is not None:
        #   h += self.Dense_0(self.act(temb))[:, :, None, None]
        
        # Add bias to each feature map conditioned on the time embedding
        if temb is not None:
            #also apply down or up to temb
            #time = self.Dense_0(self.act(temb))[:, :, None, :]
            time = torch.transpose(temb, 1,2) #[BS, timedim, frames]
            time = self.Conv_time(time) #[BS, C, frames]
            if self.time_emb_reduction == 'conv':
                h += time[:,:,None,:]
            elif self.time_emb_reduction == 'resample':
                #resample
                time = time[:,:, : :self.stride_for_time_emb]
                h += time[:,:,None,:]
        
            
        if pe_abs is not None:
            #also apply down or up to temb
            #time = self.Dense_0(self.act(temb))[:, :, None, :]
            pe_abs = self.conv_pe_abs(pe_abs) #[BS, C, frames]
            h += pe_abs



        if type(self.GroupNorm_1) == FastCumulativeGroupNorm:
            h = self.act(self.GroupNorm_1(h, train=train))
        else:
            h = self.act(self.GroupNorm_1(h))

        h = self.Dropout_0(h)

        if not self.transpose:
            h, _ = self.Conv_1(h)

            x, x_pad = self.Conv_2(x)
        else:
            h = self.Conv_1(h, input_idx=input_idx, v_pad=v_pad)

            x = self.Conv_2(x, input_idx=input_idx, v_pad=v_pad)

        h = (x + h) / np.sqrt(2.0)

        if not self.transpose:
            # print(f"ResBlockOut:{h.shape=}, {x_pad=}")
            return h, x_pad
        else:
            # print(f"ResBlockOut:{h.shape=}")
            return h


@BackboneRegistry.register("bc_ncsnpp")
class BC_NCSNpp(nn.Module):
    def add_argparse_args(parser):
        # parser.add_argument("--masking", action="store_true", help="Use masking in output layer.")
        parser.add_argument(
            "--causal", action="store_true", help="Use causal version of NCSN++_simple."
        )
        parser.add_argument("--non-causal", dest="causal", action="store_false", help="Use non-causal version of NCSN++_simple.")
        parser.set_defaults(causal=True)
        
        parser.add_argument(
            "--abs_pe", action="store_true", dest="abs_pe", help="Use causal version of NCSN++_simple."
        )
        parser.add_argument("--time_emb_reduction", type=str, default='conv', help="Do not use group norms.")

        parser.add_argument("--non_abs_pe", dest="abs_pe", action="store_false", help="Use non-causal version of NCSN++_simple.")
        parser.set_defaults(abs_pe=False)
        
        parser.add_argument(
            "--bwr", action="store_true", dest="bwr", help="Use causal version of NCSN++_simple."
        )
        parser.add_argument("--non_bwr", dest="bwr", action="store_false", help="Use non-causal version of NCSN++_simple.")
        parser.set_defaults(bwr=False)
        
        
        parser.add_argument(
            "--fir", action="store_true", dest="fir", help="Use causal version of NCSN++_simple."
        )
        parser.add_argument("--no-fir", dest="fir", action="store_false", help="Use non-causal version of NCSN++_simple.")
        parser.set_defaults(fir=True)
        
        # discrminative=True
        parser.add_argument(
            "--discriminative",
            action="store_true",
            help="Use discriminative version of NCSN++_simple.",
        )
        parser.add_argument(
            "--no_discriminative",
            action="store_false",
            dest="discriminative",
            help="Use non-discriminative version of NCSN++_simple.",
        )
        # default dscriminative=True
        # parser.set_defaults(discriminative=True)
        parser.add_argument("--dilation_freq", type=int, default=1, help="Number of ResNet Blocks per resolution. The last block performs the up/downsampling.")
        parser.add_argument("--activation", type=str, default='silu', help="Number of ResNet Blocks per resolution. The last block performs the up/downsampling.")


        parser.add_argument("--bn_freq_dim", type=int, default=4, help="Number of ResNet Blocks per resolution. The last block performs the up/downsampling.")

        # parser.add_argument("--force_native", action="store_true", help="Use native implementation for FIR filterings")
        # parser.add_argument("--force_group_norm", action="store_true", help="Use native group norm.")
        # parser.add_argument("--no_group_norm", action="store_true", help="Do not use group norms.")
        # parser.add_argument("--channels", nargs="+", default=[128, 128, 256, 256, 256, 256, 256], help="Channel dimension per resolution. The length of the array corresponds to the number of resolutions.")
        # parser.add_argument("--no-fir", action="store_false", dest="fir")
        parser.add_argument("--num_resnet_blocks", type=int, default=1, help="Number of ResNet Blocks per resolution. The last block performs the up/downsampling.")
        parser.add_argument(
            "--strides",
            nargs="+",
            type=int,
            default=[2, 2, 2, 2],
            help="Channel dimension per resolution. The length of the array corresponds to the number of resolutions.",
        )
        parser.add_argument(
            "--channels",
            nargs="+",
            type=int,
            default=[128, 256, 256, 256, 256],
            help="Channel dimension per resolution. The length of the array corresponds to the number of resolutions.",
        )
        parser.add_argument("--force_cum_group_norm", action="store_true", help="Use cumulative group norm.")

        return parser

    def __init__(
        self,
        input_channels=4,
        channels=[128, 256, 256, 256, 256],
        strides=[2, 2, 2, 2],
        out_channels=2,
        causal=True,
        force_native=False,
        force_group_norm=False,
        force_cum_group_norm=False,
        no_group_norm=False,
        fir=True,
        dilation_freq=1,
        num_resnet_blocks=1,
        time_emb_reduction='conv',
        dropout=0.0,
        discriminative=False,
        abs_pe=False,
        bwr=False,
        bn_freq_dim=4,
        activation='silu',
        # masking=False,
        **unused_kwargs,
    ):
        super().__init__()

        
        if activation == 'silu':
            act = nn.SiLU()
        elif activation == 'sigmoid':
            act = nn.Sigmoid()
        self.causal = causal
        self.force_native = force_native
        self.force_group_norm = force_group_norm
        self.force_cum_group_norm = force_cum_group_norm
        self.no_group_norm = no_group_norm
        self.fir = fir
        self.dilation_freq = dilation_freq
        self.bwr = bwr
        
        # NOTE MARIS: whatever...
        self.global_stride = int(torch.prod(torch.tensor(strides)))
        self.abs_pe = abs_pe

            
                    
        self.temp_enc = TemporalEncoding()
        self.discriminative = discriminative
        if self.discriminative:
            self.input_channels = 2  # y.real, y.imag
        else:
            self.input_channels = input_channels
        # self.masking = masking

        # input layer
        self.in_layer = Conv2d_realtime(
            self.input_channels,
            channels[0],
            kernel_size=3,
            causal=self.causal,
            n_buffs=1,
            dilation = (self.dilation_freq, 1),
        )
        

        # downsampling
        down_layers = []
        for i in range(len(channels)):
            total_stride = int(torch.prod(torch.tensor(strides[:i])))
            for j in range(num_resnet_blocks - 1):
                down_layers.append(
                    ResnetBlock(
                        in_ch=channels[i],
                        out_ch=channels[i],
                        causal=self.causal,
                        dropout=dropout,
                        fir=self.fir, act=act,
                        force_native=self.force_native,
                        force_group_norm=self.force_group_norm,
                        force_cum_group_norm=self.force_cum_group_norm,
                        no_group_norm=self.no_group_norm,
                        global_stride=self.global_stride,
                        total_stride=total_stride,
                        time_emb_reduction=time_emb_reduction, 
                        dilation_freq = self.dilation_freq,
                        pe_abs=self.abs_pe, pe_abs_ch_in=self.input_channels, ith_block=i,
                    )
                )
            if i != len(channels) - 1:  # no downsampling in the lowest resolution
                total_stride = int(torch.prod(torch.tensor(strides[: i + 1])))
                down_layers.append(
                    ResnetBlock(
                        in_ch=channels[i],
                        out_ch=channels[i + 1],
                        down=True,
                        stride=(1, strides[i]),
                        causal=self.causal,
                        dropout=dropout,
                        fir=self.fir,
                        time_emb_reduction=time_emb_reduction, 
                        force_native=self.force_native,
                        force_group_norm=self.force_group_norm,
                        force_cum_group_norm=self.force_cum_group_norm,
                        no_group_norm=self.no_group_norm,
                        global_stride=self.global_stride,
                        total_stride=total_stride, act=act,
                        dilation_freq = self.dilation_freq,
                        pe_abs=self.abs_pe, pe_abs_ch_in=self.input_channels, ith_block=i,
                    )
                )
        self.down_layers = nn.ModuleList(down_layers)

        # bottleneck
        self.bottleneck_layers = nn.ModuleList(
            [
                ResnetBlock(
                    in_ch=channels[-1],
                    out_ch=channels[-1],
                    causal=self.causal,
                    dropout=dropout,
                    fir=self.fir,
                    force_native=self.force_native,
                    time_emb_reduction=time_emb_reduction,
                    force_group_norm=self.force_group_norm,
                    force_cum_group_norm=self.force_cum_group_norm,
                    no_group_norm=self.no_group_norm, act=act,
                    global_stride=self.global_stride,
                    total_stride=self.global_stride,
                    dilation_freq = self.dilation_freq,
                    pe_abs=None, pe_abs_ch_in=self.input_channels, 
                ),
                # TODO add Attention layer?
                ResnetBlock(
                    in_ch=channels[-1],
                    out_ch=channels[-1],
                    causal=self.causal,
                    dropout=dropout,
                    fir=self.fir,
                    force_native=self.force_native,
                    time_emb_reduction=time_emb_reduction,
                    force_group_norm=self.force_group_norm,
                    force_cum_group_norm=self.force_cum_group_norm,
                    no_group_norm=self.no_group_norm,
                    global_stride=self.global_stride, act=act,
                    total_stride=self.global_stride,
                    dilation_freq = self.dilation_freq,
                    pe_abs=None, pe_abs_ch_in=self.input_channels, 
                ),
            ]
        )

        # upsampling
        up_layers = []
        for i in reversed(range(len(channels))):
            total_stride = int(torch.prod(torch.tensor(strides[:i])))
            for j in range(num_resnet_blocks - 1):
                up_layers.append(
                    ResnetBlock(
                        in_ch=channels[i] * 2,
                        out_ch=channels[i],
                        transpose=True,
                        causal=self.causal,
                        dropout=dropout,
                        fir=self.fir,
                        time_emb_reduction=time_emb_reduction,
                        force_native=self.force_native,
                        force_group_norm=self.force_group_norm,
                        force_cum_group_norm=self.force_cum_group_norm,
                        no_group_norm=self.no_group_norm,
                        global_stride=self.global_stride,
                        total_stride=total_stride, act=act,
                        dilation_freq = self.dilation_freq,
                        pe_abs=self.abs_pe, pe_abs_ch_in=self.input_channels, ith_block=i,
                    )
                )
            if i != 0:  # no upsampling in the highest resolution
                up_layers.append(
                    ResnetBlock(
                        in_ch=channels[i] * 2,
                        out_ch=channels[i - 1],
                        up=True,
                        transpose=True,
                        is_last_transposed_conv=(i == 1),
                        stride=(1, strides[i-1]),
                        causal=self.causal,
                        dropout=dropout,
                        fir=self.fir,
                        force_native=self.force_native,
                        time_emb_reduction=time_emb_reduction,
                        force_group_norm=self.force_group_norm,
                        force_cum_group_norm=self.force_cum_group_norm,
                        no_group_norm=self.no_group_norm,
                        global_stride=self.global_stride,
                        total_stride=total_stride, act=act,
                        dilation_freq = self.dilation_freq,
                        pe_abs=self.abs_pe, pe_abs_ch_in=self.input_channels, ith_block=i,
                    )
                )
        self.up_layers = nn.ModuleList(up_layers)

        # output layer
        out_layers = []
        if (self.causal and not self.force_group_norm) or self.force_cum_group_norm:
            out_layers.append(
                FastCumulativeGroupNorm(
                    num_groups=min(channels[0] // 4, 32),
                    num_channels=channels[0],
                    eps=1e-6,
                    n_buffs=self.global_stride,
                )
            )
            out_layers.append(nn.SiLU())
        elif self.no_group_norm:
            out_layers.append(nn.Identity())
            out_layers.append(nn.SiLU())
        else:
            out_layers.append(
                nn.GroupNorm(
                    num_groups=min(channels[0] // 4, 32),
                    num_channels=channels[0],
                    eps=1e-6,
                )
            )
            out_layers.append(nn.SiLU())
        out_layers.append(
            Conv2d_realtime(
                in_channels=channels[0],
                out_channels=self.input_channels,
                kernel_size=3,
                causal=self.causal,
                n_buffs=self.global_stride,
                dilation = (self.dilation_freq, 1),
            )
        )
        out_layers.append(
            Conv2d_realtime(
                in_channels=self.input_channels,
                out_channels=out_channels,
                kernel_size=1,
                causal=self.causal,
                n_buffs=self.global_stride,
                dilation = (self.dilation_freq, 1),
            )
        )
        self.out_layers = nn.ModuleList(out_layers)
        
        
        
        #absolute positional encoding
        if self.abs_pe:
            self.pe_input_layer_1 = Conv2d_realtime(
                1,
                self.input_channels,
                kernel_size=1,
                causal=self.causal,
                n_buffs=1,
                dilation = (1, 1),
            )
        
        if self.bwr:
            self.bwr_lin = torch.nn.Linear(bn_freq_dim, bn_freq_dim)

    def reset(self):
        self.in_layer.reset()

        for layer in self.down_layers:
            if type(layer) in [Conv2d_realtime, Conv2dTranspose_realtime, ResnetBlock]:
                layer.reset()

        for layer in self.bottleneck_layers:
            if type(layer) in [Conv2d_realtime, Conv2dTranspose_realtime, ResnetBlock]:
                layer.reset()

        for layer in self.up_layers:
            if type(layer) in [Conv2d_realtime, Conv2dTranspose_realtime, ResnetBlock]:
                layer.reset()

        for layer in self.out_layers:
            if type(layer) in [Conv2d_realtime, FastCumulativeGroupNorm]:
                layer.reset()

    # def forward(self, x_t, t=None, y=None, *, train=True):
    def forward(self, x_t=None, t=None, y=None, scale_divide=1, train=True):

        if t is None and y is None:
            raise ValueError('t and y cannot be empty')
            # debugging / analysis purposes only... generate some dummy data
            #t = torch.ones(1, device=x_t.device)
            #y = torch.zeros(x_t.shape, device=x_t.device) + 1j
            #in_tensor = torch.cat((x_t.real, x_t.imag, y.real, y.imag), dim=1)
            #input_index = x_t.shape[-1] - 1
        else:
            # Convert real and imaginary parts of (x,y) into X channel dimensions
            if not self.discriminative:
                input_index = x_t.shape[-1] - 1
                in_tensor = torch.cat([x_t.real, x_t.imag], dim=1)
                conditioning_tensor = torch.cat(
                    [
                        torch.cat(
                            [y[:, [in_chan], :, :].real, y[:, [in_chan], :, :].imag],
                            dim=1,
                        )
                        for in_chan in range(self.input_channels // 2 - 1)
                    ],
                    dim=1,
                )
                in_tensor = torch.cat([in_tensor, conditioning_tensor], dim=1)
            else:
                input_index = y.shape[-1] - 1
                in_tensor = torch.cat([y.real, y.imag], dim=1)

        # temporal embedding
        if t is not None:
            t_emb = self.temp_enc(t)
        else:
            t_emb = None
        # change drype to torch.float32
        if in_tensor.dtype != torch.float32:
            in_tensor = in_tensor.to(torch.float32)

        # input layer
        if self.abs_pe:
            pe_input = positionalencoding1d(in_tensor.shape[-2], self.global_stride, in_tensor)
            pe_input1, _= self.pe_input_layer_1(pe_input)
            in_tensor += pe_input
        else:
            pe_input1 = None
        
        hs = [self.in_layer(in_tensor)[0]]  # skip connections
        h = hs[-1]
        v_pads = []

        # downsampling
        for i in range(len(self.down_layers)):
            h, pad = self.down_layers[i](h, t_emb, pe_input1)
            hs.append(h)
            v_pads.append(pad)

        # bottleneck
        for i in range(len(self.bottleneck_layers)):
            h, _ = self.bottleneck_layers[i](h, t_emb)

        
        #only for BWR
        if self.bwr:
            num_frames = h.shape[-1]
            h = einops.rearrange(h, 'b c f n -> (b n) c f')
            h = self.bwr_lin(h)
            h = einops.rearrange(h, '(b n) c f -> b c f n', n = num_frames)
        
        
        
        # upsamling
        for i in range(len(self.up_layers)):
            h_skip = hs.pop()[..., -h.shape[-1] :]
            h = self.up_layers[i](
                torch.cat([h, h_skip], dim=1),
                t_emb,
                v_pad=v_pads.pop(),
                pe_abs = pe_input1,
                input_idx=input_index,
            )

        # output layers
        for i in range(len(self.out_layers)):
            if type(self.out_layers[i]) == Conv2d_realtime:
                h, _ = self.out_layers[i](h)
            elif type(self.out_layers[i]) == FastCumulativeGroupNorm:
                h = self.out_layers[i](h)
            else:
                h = self.out_layers[i](h)
        
        h = h / scale_divide
        
        # map to complex number
        h = torch.permute(h, (0, 2, 3, 1)).contiguous()

        h = torch.view_as_complex(h)[:, None, :, :]

        # print(f"ModelOut:{h.shape=}\n")

        return h
