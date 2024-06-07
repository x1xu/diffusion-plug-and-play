import math
import os
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from util.img_utils import clear_color, adjust_color
from .posterior_mean_variance import get_mean_processor, get_var_processor



__SAMPLER__ = {}

def register_sampler(name: str):
    def wrapper(cls):
        if __SAMPLER__.get(name, None):
            raise NameError(f"Name {name} is already registered!") 
        __SAMPLER__[name] = cls
        return cls
    return wrapper


def get_sampler(name: str):
    if __SAMPLER__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __SAMPLER__[name]


def create_sampler(sampler,
                   steps,
                   noise_schedule,
                   model_mean_type,
                   model_var_type,
                   dynamic_threshold,
                   clip_denoised,
                   rescale_timesteps,
                   timestep_respacing="",
                   adjust_color=False,
                   **kwargs):
    
    sampler = get_sampler(name=sampler)
    
    betas = get_named_beta_schedule(noise_schedule, steps)
    if not timestep_respacing:
        timestep_respacing = [steps]
         
    return sampler(use_timesteps=space_timesteps(steps, timestep_respacing),
                   betas=betas,
                   model_mean_type=model_mean_type,
                   model_var_type=model_var_type,
                   dynamic_threshold=dynamic_threshold,
                   clip_denoised=clip_denoised, 
                   rescale_timesteps=rescale_timesteps,
                   adjust_color=adjust_color,
                   **kwargs)


class GaussianDiffusion:
    def __init__(self,
                 betas,
                 model_mean_type,
                 model_var_type,
                 dynamic_threshold,
                 clip_denoised,
                 rescale_timesteps,
                 **kwargs
                 ):
        
        # Copying arguments for transferring to other samplers
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.dynamic_threshold = dynamic_threshold
        self.clip_denoised = clip_denoised
        self.kwargs = kwargs

        if 'adjust_color' in kwargs:
            self.adjust_color = kwargs['adjust_color']
        else:
            self.adjust_color = False
        # Other arguments will be copied later (possibly after preprocessing)


        betas = np.array(betas, dtype=np.float64)
        self.betas = betas

        # use float64 for accuracy.
        assert self.betas.ndim == 1, "betas must be 1-D"
        assert (0 < self.betas).all() and (self.betas <=1).all(), "betas must be in (0..1]"

        self.num_timesteps = int(self.betas.shape[0])
        self.rescale_timesteps = rescale_timesteps

        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.mean_processor = get_mean_processor(model_mean_type,
                                                 betas=betas,
                                                 dynamic_threshold=dynamic_threshold,
                                                 clip_denoised=clip_denoised)    
    
        self.var_processor = get_var_processor(model_var_type,
                                               betas=betas)

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        
        mean = extract_and_expand(self.sqrt_alphas_cumprod, t, x_start) * x_start
        variance = extract_and_expand(1.0 - self.alphas_cumprod, t, x_start)
        log_variance = extract_and_expand(self.log_one_minus_alphas_cumprod, t, x_start)

        return mean, variance, log_variance

    def q_sample(self, x_start, t):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        
        coef1 = extract_and_expand(self.sqrt_alphas_cumprod, t, x_start)
        coef2 = extract_and_expand(self.sqrt_one_minus_alphas_cumprod, t, x_start)

        return coef1 * x_start + coef2 * noise

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        coef1 = extract_and_expand(self.posterior_mean_coef1, t, x_start)
        coef2 = extract_and_expand(self.posterior_mean_coef2, t, x_t)
        posterior_mean = coef1 * x_start + coef2 * x_t
        posterior_variance = extract_and_expand(self.posterior_variance, t, x_t)
        posterior_log_variance_clipped = extract_and_expand(self.posterior_log_variance_clipped, t, x_t)

        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_sample_loop(self,
                      model,
                      x_start,
                      measurement,
                      measurement_cond_fn,
                      record,
                      save_root,
                      **kwargs):
        """
        The function used for sampling from noise.
        """ 
        img = x_start
        device = x_start.device

        pbar = tqdm(list(range(self.num_timesteps))[::-1])
        for idx in pbar:
            time = torch.tensor([idx] * img.shape[0], device=device)
            
            img = img.requires_grad_()
            out = self.p_sample(x=img, t=time, model=model)
            
            # Give condition.
            noisy_measurement = self.q_sample(measurement, t=time)

            # TODO: how can we handle argument for different condition method?
            img, distance = measurement_cond_fn(x_t=out['sample'],
                                      measurement=measurement,
                                      noisy_measurement=noisy_measurement,
                                      x_prev=img,
                                      x_0_hat=out['pred_xstart'])
            img = img.detach_()

            # if self.adjust_color:
            #     if (idx % 5 == 0):
            #         coef1 = extract_and_expand(self.sqrt_alphas_cumprod, idx, img)
            #         img = adjust_color(img / coef1, model, 
            #                            tol=torch.tensor([self.sqrt_one_minus_alphas_cumprod[idx]
            #                             / self.sqrt_alphas_cumprod[idx]], device=img.device)
            #                            ) * coef1
           
            pbar.set_postfix({'distance': distance.item()}, refresh=False)
            # if record:
            #     if idx % 10 == 0:
            #         file_path = os.path.join(save_root, f"progress/x_{str(idx).zfill(4)}.png")
            #         plt.imsave(file_path, clear_color(img))

        return img       
        
    def p_sample(self, model, x, t):
        raise NotImplementedError

    def p_mean_variance(self, model, x, t):
        
        model_output = model(x, self._scale_timesteps(t))
        
        # In the case of "learned" variance, model will give twice channels.
        if model_output.shape[1] == 2 * x.shape[1]:
            model_output, model_var_values = torch.split(model_output, x.shape[1], dim=1)
        else:
            # The name of variable is wrong. 
            # This will just provide shape information, and 
            # will not be used for calculating something important in variance.
            model_var_values = model_output

        model_mean, pred_xstart = self.mean_processor.get_mean_and_xstart(x, t, model_output)
        model_variance, model_log_variance = self.var_processor.get_variance(model_var_values, t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape

        return {'mean': model_mean,
                'variance': model_variance,
                'log_variance': model_log_variance,
                'pred_xstart': pred_xstart}

    
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
    
    def predict_eps_from_x_start(self, x_t, t, pred_xstart):
        coef1 = extract_and_expand(self.sqrt_recip_alphas_cumprod, t, x_t)
        coef2 = extract_and_expand(self.sqrt_recipm1_alphas_cumprod, t, x_t)
        return (coef1 * x_t - pred_xstart) / coef2

def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    elif isinstance(section_counts, int):
        section_counts = [section_counts]
    
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.
    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        if not isinstance(self, DDIMPnP) and not isinstance(self, DDPMPnP):
            self.use_timesteps = set(use_timesteps)
            self.timestep_map = []
            self.original_num_steps = len(kwargs["betas"])

            base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
            last_alpha_cumprod = 1.0
            new_betas = []
            for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
                if i in self.use_timesteps:
                    new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                    last_alpha_cumprod = alpha_cumprod
                    self.timestep_map.append(i)
            if 'timestep_map' in kwargs:
                self.timestep_map = np.array(kwargs['timestep_map'])
            kwargs["betas"] = np.array(new_betas)
            super().__init__(**kwargs)
        else:
            self.original_num_steps = len(kwargs["betas"])
            self.timestep_map = np.arange(self.original_num_steps)
            super().__init__(**kwargs)

        # else:
        #     print('right!')
        #     super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)


@register_sampler(name='ddpm')
class DDPM(SpacedDiffusion):
    def p_sample(self, model, x, t):
        out = self.p_mean_variance(model, x, t)
        sample = out['mean']

        noise = torch.randn_like(x)
        if t != 0:  # no noise when t == 0
            sample += torch.exp(0.5 * out['log_variance']) * noise

        return {'sample': sample, 'pred_xstart': out['pred_xstart']}
    
    def predict_eps_from_x_start(self, x_t, t, pred_xstart):
        coef1 = extract_and_expand(self.sqrt_recip_alphas_cumprod, t, x_t)
        coef2 = extract_and_expand(self.sqrt_recipm1_alphas_cumprod, t, x_t)
        return (coef1 * x_t - pred_xstart) / coef2
    

@register_sampler(name='ddim')
class DDIM(SpacedDiffusion):
    def p_sample(self, model, x, t, eta=0.0):
        out = self.p_mean_variance(model, x, t)
        
        eps = self.predict_eps_from_x_start(x, t, out['pred_xstart'])
        
        alpha_bar = extract_and_expand(self.alphas_cumprod, t, x)
        alpha_bar_prev = extract_and_expand(self.alphas_cumprod_prev, t, x)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )

        sample = mean_pred
        if t != 0:
            sample += sigma * noise
        
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def predict_eps_from_x_start(self, x_t, t, pred_xstart):
        coef1 = extract_and_expand(self.sqrt_recip_alphas_cumprod, t, x_t)
        coef2 = extract_and_expand(self.sqrt_recipm1_alphas_cumprod, t, x_t)
        return (coef1 * x_t - pred_xstart) / coef2


@register_sampler(name='ddim-pnp')
class DDIMPnP(SpacedDiffusion):
    
    def p_sample_loop(self,
                      model,
                      x_start,
                      measurement,
                      measurement_cond_fn,
                      record,
                      save_root,
                      **kwargs):
        
        if self.kwargs.get('warm_start', False):
            std_config = {
                "sampler": 'ddpm',
                "steps": 1000,
                "noise_schedule": 'linear',
                "model_mean_type": 'epsilon',
                "model_var_type": 'learned_range',
                "dynamic_threshold": False,
                "clip_denoised": True,
                "rescale_timesteps": False,
                "timestep_respacing": 500
            }
            default_sampler = create_sampler(adjust_color=adjust_color, **std_config) 
            default_fn = partial(
                default_sampler.p_sample_loop, 
                model=model, 
                measurement_cond_fn=kwargs['default_cond_fn'])
            img = default_fn(
                x_start=x_start, 
                measurement=measurement, 
                record=False, 
                save_root=save_root)
   
        else:
            img = x_start
        

        device = x_start.device

        denoiser = DDIMDenoiser(self)

        eta_max = kwargs.get('eta_max', 0.24)
        eta_min = kwargs.get('eta_min', 0.15)
        k = kwargs.get('k', 12)
        k_start = kwargs.get('k_start', 4)
        sigma_list = np.concatenate([
            np.repeat([eta_max], k_start),
            np.exp(np.linspace(
                np.log(eta_max), np.log(eta_min), k
            ))
        ])

        pbar = tqdm(sigma_list)
        for idx, sigma in enumerate(pbar):
            sigma_torch = torch.tensor([sigma] * img.shape[0], device=device)
            img = img.requires_grad_()
            
            
            img, distance = measurement_cond_fn(x_t=img,
                                    measurement=measurement,
                                    noisy_measurement=measurement,
                                    x_prev=img,
                                    x_0_hat=img,
                                    gamma=sigma)
            
            img = img.detach_()

            pbar.set_postfix({'distance': distance.item()}, refresh=False)
            if record:
                file_path = os.path.join(save_root, f"progress/x1_{str(idx).zfill(4)}.png")
                plt.imsave(file_path, clear_color(img))

            with torch.no_grad():
                img = denoiser.denoise(noise_lvl=sigma_torch, model=model, x_noisy=img)
                if record:
                    file_path = os.path.join(save_root, f"progress/x0_{str(idx).zfill(4)}.png")
                    plt.imsave(file_path, clear_color(img))

                if self.adjust_color:
                    if (idx == 1):
                        img = adjust_color(img, model, tol=sigma_torch)



            if idx == len(pbar) - 1:
                return img

        return img


class DDIMDenoiser:
    def __init__(self, diffuser: GaussianDiffusion):
        self.diffuser = diffuser

    def gen_schedule(self, noise_lvl):
        if isinstance(noise_lvl, torch.Tensor):
            noise_lvl = noise_lvl.item()
        time_mask = self.diffuser.alphas_cumprod > 1 / (1 + noise_lvl**2)
        masked_alphas_cumprod = self.diffuser.alphas_cumprod[time_mask]
        candidate_alphas_cumprod = ((1 + noise_lvl**2) * masked_alphas_cumprod - 1) \
                                / (noise_lvl**2 + masked_alphas_cumprod - 1)

        tol_exp = 4 * np.log(10)        # diffuse until alpha_bar < 1e-4
        max_sep = 1.1
        if noise_lvl > 0.75:
            skip = 7
        elif noise_lvl > 0.3:
            skip = 1
        else:
            skip = 0
        self.alphas_cumprod = np.empty(shape=(0,), dtype=np.float64)
        self._mapping_t = np.empty(shape=(0,), dtype=int)

        cnt_skip = skip
        for idx in range(len(candidate_alphas_cumprod)):
            if cnt_skip < skip:
                cnt_skip += 1
                continue
            this = candidate_alphas_cumprod[idx]
            if this < np.exp(-tol_exp):
                # already good
                self.alphas_cumprod = np.append(self.alphas_cumprod, this)
                self._mapping_t = np.append(idx, self._mapping_t)
                break

            if idx == len(candidate_alphas_cumprod) - 1:
                # reaching the end but not good yet
                # append more iterations at the end
                n_to_append = int((np.log(this) + tol_exp) / np.log(max_sep)) + 1
                alphas_bar_to_append = np.exp(
                    np.linspace(np.log(this), -tol_exp, n_to_append)
                )
                self.alphas_cumprod = np.concatenate(
                    (self.alphas_cumprod, alphas_bar_to_append))
                self._mapping_t = np.concatenate(
                    (self._mapping_t, np.repeat([idx], n_to_append)))
                break

            next = candidate_alphas_cumprod[idx + 1]
            if this >= max_sep * next:
                # too large gap, do interpolation
                n_to_append = int(np.log(this / next) / np.log(max_sep)) + 1
                alphas_bar_to_append = np.exp(
                    np.linspace(np.log(this), np.log(next), n_to_append + 1)
                )[:-1]        # excluding `next`
                self.alphas_cumprod = np.append(self.alphas_cumprod, 
                                                alphas_bar_to_append)
                self._mapping_t = np.concatenate(
                    (self._mapping_t, 
                    np.repeat([idx], n_to_append//2), 
                    np.repeat([idx+1], n_to_append - n_to_append//2)))
            else:
                self.alphas_cumprod = np.append(self.alphas_cumprod, this)
                self._mapping_t = np.append(self._mapping_t, idx)
            
            if len(candidate_alphas_cumprod) - idx > 15:
                cnt_skip = 0
        
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)

        self.coefs = (noise_lvl**2 - 1) * self.alphas_cumprod + 1
        self.sqrt_coefs = np.sqrt(self.coefs)
        self.sqrt_coefs_prev = np.append(noise_lvl, self.sqrt_coefs[:-1])
        self.coefs_w = self.aux_w(noise_lvl, self.alphas_cumprod)
        self.coefs_w_prev = self.aux_w(noise_lvl, self.alphas_cumprod_prev)
    def mapping_t(self, t):
        return self._mapping_t[t]

    def mapping_t_x(self, noise_lvl, t, x, x_noisy):
        t_uncond = self.mapping_t(t)
        sqrt_alpha_bar_uncond = extract_and_expand(
            self.diffuser.sqrt_alphas_cumprod,
            t_uncond,
            x)
        sqrt_alpha_bar = extract_and_expand(
            self.sqrt_alphas_cumprod, t, x
        )
        coef = extract_and_expand(
            self.coefs, t, x
        )
        noise_lvl = noise_lvl.float()

        x_uncond = sqrt_alpha_bar_uncond * x_noisy \
                   + noise_lvl**2 * sqrt_alpha_bar * sqrt_alpha_bar_uncond * x / coef
        return t_uncond, x_uncond
    
    def _tau_prime_do_not_use(alpha_bar, noise_lvl):
        return .5 * np.log(
            1 + noise_lvl**2 * (1 - alpha_bar) / (1 + (noise_lvl**2-1) * alpha_bar)
        )

    def denoise(self, noise_lvl, model, x_noisy, eta=0.0):
        x = torch.randn_like(x_noisy)
        self.gen_schedule(noise_lvl)

        for t in list(range(len(self.alphas_cumprod)))[::-1]:
            t_uncond, x_uncond = self.mapping_t_x(noise_lvl, t, x, x_noisy)
            t_uncond = torch.tensor([t_uncond], device=noise_lvl.device)
            out = self.diffuser.p_mean_variance(model, x_uncond, t_uncond)
            eps = self.diffuser.predict_eps_from_x_start(x_uncond, t_uncond, out['pred_xstart'])

            sqrt_coef = extract_and_expand(self.sqrt_coefs, t, x)
            sqrt_coef_prev = extract_and_expand(self.sqrt_coefs_prev, t, x)
            coef_w = extract_and_expand(self.coefs_w, t, x)
            coef_w_prev = extract_and_expand(self.coefs_w_prev, t, x)
            
            x = sqrt_coef_prev / sqrt_coef * x \
                - sqrt_coef_prev * (coef_w - coef_w_prev) * eps

        return x + x_noisy

    def aux_u(self, noise_lvl, alpha_bar):
        return -1 / np.sqrt(noise_lvl**2 - 1 + 1/alpha_bar)
    
    def aux_v(self, noise_lvl, alpha_bar):
        return -np.arctan(
            noise_lvl**2 / np.sqrt((noise_lvl**2 + 1) / alpha_bar - 1))

    def aux_w(self, noise_lvl, alpha_bar):
        is_degen = (1 - alpha_bar) < 1e-3
        result = np.empty_like(alpha_bar)
        result[is_degen] = -np.pi/2 + np.arctan(np.sqrt(1/alpha_bar[is_degen] - 1) / noise_lvl)
        result[~is_degen] = -np.arctan(noise_lvl / np.sqrt(1/alpha_bar[~is_degen] - 1))
        return result


@register_sampler(name='ddpm-pnp')
class DDPMPnP(SpacedDiffusion):
    
    def p_sample_loop(self,
                      model,
                      x_start,
                      measurement,
                      measurement_cond_fn,
                      record,
                      save_root,
                      **kwargs):
        
        if self.kwargs.get('warm_start', False):
            std_config = {
                "sampler": 'ddpm',
                "steps": 1000,
                "noise_schedule": 'linear',
                "model_mean_type": 'epsilon',
                "model_var_type": 'learned_range',
                "dynamic_threshold": False,
                "clip_denoised": True,
                "rescale_timesteps": False,
                "timestep_respacing": 500
            }
            default_sampler = create_sampler(adjust_color=adjust_color, **std_config) 
            default_fn = partial(
                default_sampler.p_sample_loop, 
                model=model, 
                measurement_cond_fn=kwargs['default_cond_fn'])
            img = default_fn(
                x_start=x_start, 
                measurement=measurement, 
                record=False, 
                save_root=save_root)
   
        else:
            img = x_start
        
        device = x_start.device

        denoiser = DDPMDenoiser(self)

        eta_max = kwargs.get('eta_max', 0.24)
        eta_min = kwargs.get('eta_min', 0.15)
        k = kwargs.get('k', 12)
        k_start = kwargs.get('k_start', 4)
        sigma_list = np.concatenate([
            np.repeat([eta_max], k_start),
            np.exp(np.linspace(
                np.log(eta_max), np.log(eta_min), k
            ))
        ])

        pbar = tqdm(sigma_list)
        for idx, sigma in enumerate(pbar):
            
            sigma_torch = torch.tensor([sigma] * img.shape[0], device=device)
            with torch.no_grad():
                img = denoiser.denoise(noise_lvl=sigma_torch, model=model, x_noisy=img)
                if record:
                # if idx % 10 == 0:
                    file_path = os.path.join(save_root, f"progress/x0_{str(idx).zfill(4)}.png")
                    plt.imsave(file_path, clear_color(img))

            if self.adjust_color:
                if (idx == 2):
                    img = adjust_color(img, model, tol=sigma_torch)
            
            if idx == len(pbar) - 1:
                return img
            
            img = img.requires_grad_()

            
            img, distance = measurement_cond_fn(x_t=img,
                                      measurement=measurement,
                                      noisy_measurement=measurement,
                                      x_prev=img,
                                      x_0_hat=img,
                                      gamma=sigma)
            img = img.detach_()
           
            pbar.set_postfix({'distance': distance.item()}, refresh=False)
            if record:
                # if idx % 10 == 0:
                file_path = os.path.join(save_root, f"progress/x1_{str(idx).zfill(4)}.png")
                plt.imsave(file_path, clear_color(img))

        return img  


class DDPMDenoiser:
    def __init__(self, diffuser: GaussianDiffusion):
        self.diffuser = diffuser

    def gen_schedule(self, noise_lvl):
        if isinstance(noise_lvl, torch.Tensor):
            noise_lvl = noise_lvl.item()
        time_mask = self.diffuser.alphas_cumprod > 1 / (1 + noise_lvl**2)
        masked_alphas_cumprod = self.diffuser.alphas_cumprod[time_mask]
        
        self.ts = 1. / masked_alphas_cumprod - 1.
        self.ts_prev = np.append(0., self.ts[:-1])

        self.sqrt_ts = np.sqrt(self.ts)
        self.sqrt_ts_prev = np.sqrt(self.ts_prev)

        self.sqrt_diff_ts = np.sqrt(self.ts - self.ts_prev)

    def mapping_t(self, t):
        return t

    def mapping_t_x(self, noise_lvl, t, x, x_noisy):
        t_uncond = self.mapping_t(t)
        sqrt_alpha_bar_uncond = extract_and_expand(
            self.diffuser.sqrt_alphas_cumprod,
            t_uncond,
            x)
        x_uncond = sqrt_alpha_bar_uncond * x
        return t_uncond, x_uncond

    def denoise(self, noise_lvl, model, x_noisy, eta=0.0):
        x = x_noisy.detach().clone()
        noise_lvl = noise_lvl.float()
        self.gen_schedule(noise_lvl)

        for t in list(range(len(self.ts)))[::-1]:
            t_uncond, x_uncond = self.mapping_t_x(noise_lvl, t, x, x_noisy)
            t_uncond = torch.tensor([t_uncond], device=noise_lvl.device)
            out = self.diffuser.p_mean_variance(model, x_uncond, t_uncond)
            eps = self.diffuser.predict_eps_from_x_start(x_uncond, t_uncond, out['pred_xstart'])

            sqrt_t = extract_and_expand(self.sqrt_ts, t, x)
            sqrt_t_prev = extract_and_expand(self.sqrt_ts_prev, t, x)
            sqrt_diff_t = extract_and_expand(self.sqrt_diff_ts, t, x)
            
            w = torch.randn_like(x)
            x += 2 * (sqrt_t_prev - sqrt_t) * eps
            if t != 0:
                x += sqrt_diff_t * w

        return x


# =================
# Helper functions
# =================

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

# ================
# Helper function
# ================

def extract_and_expand(array, time, target):
    array = torch.from_numpy(array).to(target.device)[time].float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)


def expand_as(array, target):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array)
    elif isinstance(array, np.float):
        array = torch.tensor([array])
   
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)

    return array.expand_as(target).to(target.device)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def denoise_simple(img, model, noise_lvl):
    scale = 1 / torch.sqrt(noise_lvl**2 + 1).float()
    rescaled_img = img * scale
    t = (1 - scale) / 0.0002    # This is really an arbitrary choice
    t = t.float()
    pred_eps = model(rescaled_img, t.float())
    return img - t * pred_eps