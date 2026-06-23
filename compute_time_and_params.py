from numpy import mean
import torch
from torch.version import cuda
from sgmse.backbones.blockcausal_ncsnpp import BC_NCSNpp
from torch.utils.flop_counter import FlopCounterMode
import numpy as np


class NCSNppWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        model.eval()
        #self.model = torch.compile(self.model, dynamic=False)
        
    def forward(self, x):
        # Create dummy tensors for the required arguments
        batch_size = x.shape[0]
        time_cond = torch.ones(batch_size, 1, device=x.device)  # shape: (batch_size, 1)
        #time_cond = None
        y = torch.zeros_like(x) + 1j  # same shape as x
        x = x + 1j
        scale_divide = torch.ones(batch_size, 1, 1, 1, device=x.device)  # shape: (batch_size, 1, 1, 1)
        
        return self.model(x, time_cond, y, scale_divide)



def get_flops(model, inp, with_backward=False):
    
    istrain = model.training
    model.eval()
    
    inp = inp if isinstance(inp, torch.Tensor) else torch.randn(inp)

    flop_counter = FlopCounterMode(mods=model, display=False, depth=None)
    with flop_counter:
        if with_backward:
            model(inp).sum().backward()
        else:
            model(inp)
    total_flops =  flop_counter.get_total_flops()
    if istrain:
        model.train()
    return total_flops



if __name__ == '__main__':
    channels=[128, 256, 256, 256, 256]
    strides=[2, 2, 2, 4]
    
    nf = 96
    ch_mult = (1, 2, 2, 2)
    num_res_blocks = 1
    model = BC_NCSNpp(ch_mult=ch_mult, nf = nf, num_res_blocks=num_res_blocks).cuda() 
    
   # model = torch.compile(model, dynamic=False)
    wrapped_model = NCSNppWrapper(model).cuda()
    x = torch.randn(1, 1, 256, 64).cuda()
    a = get_flops(wrapped_model, x, with_backward=False)
    print(f"FLOPs (forward only): {a:_}".replace("_", "."))
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number of trainable params {params}')
    
    #with torch.cuda.device(0):
    #    net = wrapped_model
    #    flops, params = get_model_complexity_info(net, (1, 256, 64), as_strings=True,
    #                                                print_per_layer_stat=True, verbose=True)
    #    
    #    print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    #    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    #Run model and measure time

    x = torch.randn(1, 1, 256, 64).cuda()
    batch_size = x.shape[0]
    time_cond = torch.ones(batch_size, 1, device=x.device)  # shape: (batch_size, 1)
    #time_cond = None
    y = torch.zeros_like(x) + 1j  # same shape as x
    x = x + 1j
    scale_divide = torch.ones(batch_size, 1, 1, 1, device=x.device)  # shape: (batch_size, 1, 1, 1)
        
    model(x, time_cond, y, scale_divide)
    model(x, time_cond, y, scale_divide)
    model(x, time_cond, y, scale_divide)
    stat = []
    for i in range(100):
        with torch.no_grad():
            start = cuda.Event(enable_timing=True)
            model(x, time_cond, y, scale_divide)
            end = cuda.Event(enable_timing=True)
            stat.append(start.elapsed_time(end)/1000)
    print(f"Time taken for 100 iterations: {mean(stat)} seconds")