from abc import ABC, abstractmethod
import numpy as np
import torch

__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)

    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if hasattr(self.operator, 'dither'):
            # In this case the measurement operator is fixed
            yx = measurement * x_0_hat
            loss = torch.mean(
                torch.log(1. + torch.exp(-yx / self.operator.intensity)))
            norm = loss
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        
        elif self.noiser.__name__ == 'gaussian':
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm
    
    def sq_grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if hasattr(self.operator, 'dither'):
            # In this case the measurement operator is fixed
            yx = measurement * x_0_hat
            loss = torch.sum(
                torch.log(1. + torch.exp(-yx / self.operator.intensity)))
            sq_norm = loss
            sq_norm_grad = torch.autograd.grad(outputs=sq_norm, inputs=x_prev)[0]
            # sq_norm /= sq_norm
        
        elif self.noiser.__name__ == 'gaussian':
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            sq_norm = torch.sum(difference ** 2)
            sq_norm_grad = torch.autograd.grad(outputs=sq_norm, inputs=x_prev)[0]

        
        # elif self.noiser.__name__ == 'poisson':
        #     Ax = self.operator.forward(x, **kwargs)
        #     difference = measurement-Ax
        #     norm = torch.linalg.norm(difference) / measurement.abs()
        #     norm = norm.mean()
        #     norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]

        else:
            raise NotImplementedError
             
        return sq_norm_grad, sq_norm
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t
    
@register_conditioning_method(name='projection')
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name='mcg')
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        
        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm
        
@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        with torch.no_grad():
            x_t -= norm_grad * self.scale
        return x_t, norm
        
@register_conditioning_method(name='ps+')
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get('num_sampling', 5)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            # TODO: use noiser?
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling
        
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm
    
@register_conditioning_method(name='lgd')
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get('num_sampling', 5)
        self.scale = kwargs.get('scale', 1.0)
        self.lbda = kwargs.get('lambda', 1e-3)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        exp_neg_sum = 0.
        for i in range(self.num_sampling):
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            if hasattr(self.operator, 'dither'):
                # In this case the measurement operator is fixed
                yx = measurement * x_0_hat_noise
                loss = torch.sum(
                    torch.log(1. + torch.exp(-yx / self.operator.intensity)))
                sq_norm = loss
            else:
                difference = measurement - self.operator.forward(x_0_hat_noise)
                sq_norm = torch.sum(difference ** 2)

            exp_neg_sum += torch.exp(-sq_norm * self.lbda)

        norm = torch.sqrt(-torch.log(exp_neg_sum / self.num_sampling))

        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm

@register_conditioning_method(name='inpainting_prox')
class ProxSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        meas_noise_lvl = 0.15
        gamma = kwargs['gamma']
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        with torch.no_grad():
            x_meas = self.operator.forward(x_0_hat, **kwargs)
            x_orth_proj = x_0_hat - x_meas
            x_t = x_orth_proj \
                  + measurement / (1 + (meas_noise_lvl/gamma)**2) \
                  + x_meas / (1 + (gamma/meas_noise_lvl)**2)
            perturb = torch.randn_like(x_t)
            perturb_meas = self.operator.forward(perturb, **kwargs)
            perturb_orth_proj = perturb - perturb_meas
            # May improve performance in some cases
            # gamma_clipped = np.min((0.5, gamma))
            gamma_clipped = gamma
            x_t += meas_noise_lvl * gamma_clipped / (meas_noise_lvl**2 + gamma_clipped**2) ** 0.5 \
                   * perturb_meas \
                   + gamma_clipped * perturb_orth_proj

        return x_t, norm
    
@register_conditioning_method(name='mala')
class MALASampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 0.5)
        self.reg = kwargs.get('reg', 50.0)
        self.meas_noise_lvl = kwargs.get('multiplier', 0.15)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        meas_noise_lvl = self.meas_noise_lvl
        gamma = kwargs['gamma']
        x0 = x_prev.detach().clone()
        x = x_t.detach().clone()
        
        accepted = True
        scale_normalizer = np.min((meas_noise_lvl**2, gamma**2))
        stepsize = self.scale * scale_normalizer
        coef_x0 = 1 - np.exp(-stepsize / gamma**2)
        coef_v = gamma**2 * (1 - np.exp(-stepsize / gamma**2))
        coef_b = gamma * np.sqrt(1 - np.exp(-2 * stepsize / gamma**2))

        for i in range(1000):

            x = x.requires_grad_(True)

            if accepted:
                grad_x, norm_x = self.sq_grad_and_value(x_prev=x, x_0_hat=x, measurement=measurement, **kwargs)
            
            x = x.detach_()

            stepsize = self.scale * scale_normalizer / (self.reg + torch.sqrt(norm_x))
            coef_x0 = 1 - torch.exp(-stepsize / gamma**2)
            coef_v = gamma**2 * (1 - torch.exp(-stepsize / gamma**2))
            coef_b = gamma * torch.sqrt(1 - torch.exp(-2 * stepsize / gamma**2))

            w = torch.randn_like(x).detach_()

            with torch.no_grad():
                x += coef_x0 * (x0 - x) \
                    - coef_v * grad_x / 2 / meas_noise_lvl ** 2 \
                    + coef_b * w
 
            """ The Metropolis-adjustment step. Can be omitted occasionally. """
            # with torch.no_grad():
            #     x_prime = (1 - coef_x0) * x + coef_x0 * x0 \
            #             - coef_v * grad_x / 2 / meas_noise_lvl ** 2 \
            #             + coef_b * w
            
            # x_prime = x_prime.requires_grad_(True)

            # grad_x_prime, norm_x_prime = self.sq_grad_and_value(
            #     x_prev=x_prime, x_0_hat=x_prime, measurement=measurement, **kwargs)
            
            # x_prime = x_prime.detach_()

            # with torch.no_grad():
            #     log_pi_x = -norm_x / 2 / meas_noise_lvl**2 \
            #                - torch.sum((x - x0) ** 2) / 2 / gamma**2
            #     log_pi_x_prime = -norm_x_prime / 2 / meas_noise_lvl**2 \
            #                      - torch.sum((x_prime - x0) ** 2) / 2 / gamma**2
            #     log_Q_x_and_x_prime = -torch.sum(w ** 2) / 2 / coef_b**2
            #     log_Q_x_prime_and_x = -torch.sum(
            #         (x - ((1 - coef_x0) * x_prime + coef_x0 * x0
            #                 - coef_v * grad_x_prime / 2 / meas_noise_lvl**2)
            #         ) ** 2
            #     ) / 2 / coef_b**2
            #     # accept_prob = pi_x_prime * Q_x_prime_and_x / pi_x / Q_x_and_x_prime
            #     accept_prob = torch.exp(
            #         log_pi_x_prime + log_Q_x_prime_and_x
            #         - log_pi_x - log_Q_x_and_x_prime)
            #     toss = torch.rand(1, device=accept_prob.device)
            #     if toss <= accept_prob:
            #         x = x_prime
            #         accepted = True
            #     else:
            #         accepted = False
            
            x = x.detach_()
        
        return x, norm_x
