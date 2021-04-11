import numpy as np
from glob import glob

import torch
torch.manual_seed(0)

import matplotlib.pyplot as plt


def plot_vae_result(x, vae, device):
    h, w = x.shape[-2:]
    with torch.no_grad():
        predict_x = vae(x.reshape(1, 3, h, w).to(device))[0].to('cpu')

    init_image = (x.detach().numpy().reshape(h, w, 3) + 1) / 2
    predict_image = (predict_x.detach().numpy().reshape(h, w, 3) + 1) / 2

    plt.imshow(np.hstack((init_image, predict_image)))
    
def img1_convert_to_img2(x1, x2, vae, device):
    h, w = x1.shape[-2:]
    with torch.no_grad():
        z1 = vae.encode(x1.reshape(1, 3, h, w).to(device))[0]
        z2 = vae.encode(x2.reshape(1, 3, h, w).to(device))[0]

    intermediate_images = np.zeros((9, h, w, 3))
    for i in range(9):
        z = ((8 - i) * z1 +  i * z2) / 8
        intermediate_images[i] = (vae.decode(z).to('cpu').detach().numpy().reshape(h, w, 3) + 1) / 2

    plt.imshow(np.hstack(intermediate_images))