import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as TF
from torchvision.utils import make_grid
from IPython.display import display, HTML, clear_output
from helpfunction import get , frames2vid




@torch.inference_mode()
def reverse_diffusion(model, sd, h1, timesteps=1000, img_shape=(2, 16, 64),
                      num_images=1, nrow=8, device="cpu", **kwargs):
    save_path = kwargs.get('save_path', None)


    h2_noisy = torch.randn((num_images, *img_shape), device=device)
    h1 = h1.to(device)


    model.eval()





    for time_step in tqdm(iterable=reversed(range(1, timesteps)),
                          total=timesteps-1, dynamic_ncols=False,
                          desc="Sampling :: ", position=0):

        ts = torch.ones(num_images, dtype=torch.long, device=device) * time_step
        z = torch.randn_like(h2_noisy) if time_step > 1 else torch.zeros_like(h2_noisy)



        predicted_noise = model(h1, h2_noisy, ts)


        beta_t = get(sd.beta, ts)
        one_by_sqrt_alpha_t = get(sd.one_by_sqrt_alpha, ts)
        sqrt_one_minus_alpha_cumulative_t = get(sd.sqrt_one_minus_alpha_cumulative, ts)


        h2_noisy = (
            one_by_sqrt_alpha_t
            * (h2_noisy - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise)
            + torch.sqrt(beta_t) * z
        )



















    h2_noisy = h2_noisy.detach().cpu().numpy()






    np.save(save_path, h2_noisy)
    return h2_noisy

