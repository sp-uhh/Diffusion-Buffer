import torch
from typing import List, Callable

from sgmse.backbones.blockcausal_ncsnpp import BC_NCSNpp

def receptive_time_indices(
    x: torch.Tensor,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    freq_index = 0,          # which frequency row to probe
) -> List[List[int]]:
    """
    For a model that maps a (1,1,F,T) tensor to another (…,F',T') tensor,
    find, **for each output time step**, the set of *input-time* positions
    that affect it.

    Parameters
    ----------
    x           : input (will be cloned & `requires_grad=True`)
    forward_fn  : your network; may return a tensor or a tuple/list
    freq_index  : which frequency bin to pick when extracting a scalar
                  (default 0 ⇒ first row).  Pass None to SUM over freq.

    Returns
    -------
    mapping : list of length T'  
              mapping[t_out] → zero-based input-time indices
    """
    # 1. rebuild the computational graph
    x0 = x.detach().clone().requires_grad_(True)
    t = torch.randn(1, 64)

    out = forward_fn(x0, t, x0)
    if isinstance(out, (tuple, list)):
        out = out[0]                    # keep the last output

    probe = out.real if torch.is_complex(out) else out
    N, C, F_out, T_out = probe.shape

    if freq_index is None:
        # aggregate all freq rows into one scalar per time step
        selector = lambda t: probe[0, 0, :, t].sum()
    else:
        if not (-F_out <= freq_index < F_out):
            raise ValueError(f"freq_index {freq_index} out of range (0…{F_out-1})")
        selector = lambda t: probe[0, 0, freq_index, t]

    mapping = []
    for t in range(T_out):
        if x0.grad is not None:
            x0.grad.zero_()

        selector(t).backward(retain_graph=True)

        # collapse channel & freq → keep only the time dimension of the input
        g = x0.grad.squeeze()            # now shape (F_in, T_in)
        time_idxs = g.any(dim=0).nonzero(as_tuple=True)[0].tolist()
        mapping.append(time_idxs)

    # 2. rip out phantom / duplicate outputs (left ghost after TConv, etc.)
    #    – keep the right-most copy.
    cleaned = []
    for seg in mapping:
        if not seg or max(seg) >= x0.shape[-1]:
            continue                     # drops pure-padding outputs
        if not cleaned or seg != cleaned[-1]:
            cleaned.append(seg)
        else:
            cleaned[-1] = seg            # overwrite → keep right copy

    return cleaned, mapping


if __name__=="__main__":
    network = BC_NCSNpp(causal=True, discriminative=True,
                                                          strides=[2, 2, 2, 4], 
                                                          channels=[128, 256, 256, 256, 256])
    
    # Example how to call model
    num_frames = 64
    num_freq = 256
    x = torch.randn(1, 1, num_freq, num_frames, dtype=torch.complex64) #input
    t = torch.randn(1, num_frames) #time embedding
    output  = network(x, t, x)  #xt, t, y

    cleaned_output, all_outputs = receptive_time_indices(x, network)

    print("x → y:")
    for j, seg in enumerate(all_outputs, 1):
        # indices = ",".join(str(i+1) for i in seg)    # 1-based for display
        indices = ",".join([str(seg[0]+1), str(seg[-1]+1)])
        print(f" out[{j}] ← x[{indices}]")