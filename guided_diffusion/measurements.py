'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
from torch.nn import functional as F
from torchvision import torch

from util.resizer import Resizer
from util.img_utils import fft2_m, ifft2_m


# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
    
    def forward(self, data):
        return data

    def transpose(self, data):
        return data
    
    def ortho_project(self, data):
        return data

    def project(self, data):
        return data


@register_operator(name='super_resolution')
class SuperResolutionOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1/scale_factor).to(device)

    def forward(self, data, **kwargs):
        return self.down_sample(data)

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)

class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 

@register_operator(name='ordinary_phase_retrieval')
class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device
        
    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude

    def fft(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        fft = fft2_m(padded)
        return fft

    def ifft(self, f, **kwargs):
        padded = ifft2_m(f)
        ifft_f = padded[..., self.pad:-self.pad, self.pad:-self.pad]
        return ifft_f
    
    def get_mask(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        mask = torch.zeros_like(padded, dtype=torch.bool)
        mask[..., self.pad:-self.pad, self.pad:-self.pad] = True
        return mask

    def fft_full(self, data, **kwargs):
        return fft2_m(data)
    
    def ifft_full(self, f, **kwargs):
        padded = ifft2_m(f)
        return padded
    
    def truncate(self, data, **kwargs):
        return data[..., self.pad:-self.pad, self.pad:-self.pad]
    


@register_operator(name='coded_pr')
class CodedPROperator(NonLinearOperator):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device
        mod_gen = torch.randint(0, 8, size=(256, 256), device=self.device)
        self.modulation = -(mod_gen == 0).long() + (mod_gen == 7).long()
        
    def modulate(self, data, **kwargs):
        return data * self.modulation.expand_as(data)

    def forward(self, data, **kwargs):
        padded = F.pad(self.modulate(data), (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude

    def fft(self, data, **kwargs):
        padded = F.pad(self.modulate(data), (self.pad, self.pad, self.pad, self.pad))
        fft = fft2_m(padded)
        return fft

    def ifft(self, f, **kwargs):
        padded = ifft2_m(f)
        ifft_f = padded[..., self.pad:-self.pad, self.pad:-self.pad]
        return ifft_f
    
    def get_mask(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        mask = torch.zeros_like(padded, dtype=torch.bool)
        mask[..., self.pad:-self.pad, self.pad:-self.pad] = True
        return mask
    
    def fft_full(self, data, **kwargs):
        return fft2_m(data)
    
    def ifft_full(self, f, **kwargs):
        padded = ifft2_m(f)
        return padded
    
    def truncate(self, data, **kwargs):
        return data[..., self.pad:-self.pad, self.pad:-self.pad]
    


@register_operator(name='quantization')
class QuantizationOperator(NonLinearOperator):
    def __init__(self, dither_name, intensity, **kwargs):
        # TODO: support multiple trials
        # TODO: support other dithering distribution
        self.dither = dither_name
        if not (dither_name == 'logit'):
            raise NotImplementedError(f"Dithering {dither_name} unsupported")
        self.intensity = intensity
        
    def forward(self, data, **kwargs):
        random = torch.rand_like(data)
        p = 1 / (1 + torch.exp(-data / self.intensity))
        return 2.0 * (random < p) - 1.0

# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson
       
        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

    
        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0
       
        # return data.clamp(low_clip, 1.0)