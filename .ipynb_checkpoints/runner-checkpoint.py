import sys, os, torch
sys.path.append(".")
sys.path.append('./taming-transformers')
from taming.models import vqgan # checking correct import from taming

import numpy as np
import IPython.display as d
from PIL import Image

from notebook_helpers import get_model
from notebook_helpers import run

import warnings 
warnings.filterwarnings('ignore') 

mode= 'superresolution'
model = get_model(mode)

custom_steps = 100
attackldm_path = "/home/cwtang/deepmind-research/adversarial_robustness/pytorch/"

def runner(image_idx):
    print("|- Runner")
    image_path = os.path.join(attackldm_path, "tmp_cifar10/image{}.jpg".format(image_idx))
    logs = run(model["model"], image_path, mode, custom_steps)
    
    # Process the output
    sample = logs["sample"]
    sample = sample.detach().cpu()
    sample = torch.clamp(sample, -1., 1.)
    sample = (sample + 1.) / 2. * 255
    sample = sample.numpy().astype(np.uint8)
    sample = np.transpose(sample, (0, 2, 3, 1))
    
    # Convert to PIL Image
    processed_image = Image.fromarray(sample[0])

    # Read original image to get its dimensions
    original_image = Image.open(image_path)
    orig_width, orig_height = original_image.size

    # Resize the processed image to original dimensions
    resized_image = processed_image.resize((orig_width, orig_height),)

    resized_image.save(os.path.join(attackldm_path, "cifar10_after_runner/image{}.jpg".format(image_idx))) 
    print("|- Runner done")

if __name__ == '__main__':
    image_idx = sys.argv[1]
    runner(int(image_idx))