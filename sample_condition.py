from functools import partial
import os
import argparse
import yaml
import shutil

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import lpips
from skimage.metrics import peak_signal_noise_ratio

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator, prepare_im
from util.logger import get_logger

from datetime import datetime


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
   
    #assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    #"learn_sigma must be the same for model and diffusion configuartion."
    
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    use_warm_start = 'initializer' in cond_config
    if use_warm_start:
        initializer = cond_config['initializer']
        default_cond_method = get_conditioning_method(
            initializer['method'], operator, noiser, **initializer['params'])
    else:
        default_cond_method = cond_method
    default_cond_fn = default_cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
    logger.info(f"Diffusion solver : {diffusion_config['sampler']}")

    if 'nonunique_color' in cond_config:
        adjust_color = cond_config['nonunique_color']
    else:
        adjust_color = False

    # Load diffusion sampler
    print(use_warm_start)
    sampler = create_sampler(adjust_color=adjust_color, warm_start=use_warm_start, **diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn,
                        **diffusion_config.get('params', {}))
   
    # Working directory
    out_path = os.path.join(args.save_dir, datetime.now().strftime('%m-%d/%H:%M'), measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    batch_size = 1  # Do not change this value. Larger batch size is not available for particle size > 1.
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.CenterCrop((256, 256)),
                                    transforms.Resize((256, 256)),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # Preprocessing shared by FFHQ and ImageNet.
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=batch_size, num_workers=0, train=False)

    if 'init_config' in data_config:
        initializer = get_dataloader(
            get_dataset(**data_config['init_config'], transforms=transform),
            batch_size=batch_size, num_workers=0, train=False
        )
        initializer = iter(initializer)

    start_at = data_config.get('start_at', 0)

    # (Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )

    shutil.copy(args.model_config, out_path)
    shutil.copy(args.diffusion_config, out_path)
    shutil.copy(args.task_config, out_path)
        
    # Do Inference
    for i, ref_img in enumerate(loader):
        if i < start_at:
            x_start = next(initializer)
            continue
        
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + diffusion_config['sampler'] + '.png'
        ref_img = ref_img.to(device)

        # Exception) In case of inpainging,
        if measure_config['operator'] ['name'] == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=1)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            default_cond_fn = partial(default_cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)
            y_n = operator.forward(ref_img, mask=mask)

        else: 
            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img)
            y_n = noiser(y)
         
        # Sampling
        if 'init_config' in data_config:
            x_start = next(initializer).to(device)
        else:
            x_start = torch.randn(ref_img.shape, device=device).requires_grad_()

        sample = sample_fn(
            x_start=x_start, 
            measurement=y_n, 
            default_cond_fn=default_cond_fn, 
            record=True, 
            save_root=out_path
        )

        psnr_normal = peak_signal_noise_ratio(ref_img.detach().cpu().numpy(), sample.detach().cpu().numpy())

        plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
        plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
        plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))

        prepared_ref_img = prepare_im(os.path.join(out_path, 'label', fname), 256, device=device)
        prepared_recon = prepare_im(os.path.join(out_path, 'recon', fname), 256, device=device)

        normal_d = loss_fn_vgg(prepared_recon, prepared_ref_img)

        print('LPIPS: ', normal_d.item(),
            'Reconstruction MSE: ', psnr_normal)




if __name__ == '__main__':
    main()
