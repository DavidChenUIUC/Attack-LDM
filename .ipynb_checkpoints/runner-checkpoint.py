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
dir_ = './data/example_conditioning/superresolution'
cond_choice = os.listdir(dir_)

for im in cond_choice:
    if "jpg" not in im and "png" not in im and "jpeg" not in im: continue

    if 'img1' in im or 'celeba' in im : pass
    else: continue
    
    
    # Save the resized image
    im_name = im.split('.')[0]
    storing_name = './attacked_imgs/'+im_name + "_attacked-ratio3-0.4.jpg"

    #################################
    
    print("|- Running: ", storing_name)
    cond_choice_path = os.path.join(dir_, im)    
    
    logs = run(model["model"], cond_choice_path, mode, custom_steps)
    
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
    original_image = Image.open(cond_choice_path)
    orig_width, orig_height = original_image.size

    # Resize the processed image to original dimensions
    resized_image = processed_image.resize((orig_width, orig_height),)

    
    resized_image.save(storing_name)
