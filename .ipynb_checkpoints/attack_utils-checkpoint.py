from torchattacks import PGD
import torchattacks, torch
from torchvision import models as torchvision_models
import torch
torch.manual_seed(0)
torch.cuda.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.enable_grad()
def apply_attack(model, images, labels):
    ratio = 1
    # atk = PGD(model, eps=8/255, alpha=2/225, steps=10, random_start=True)
    atk1 = torchattacks.FGSM(model, eps=(8/255* ratio))
    atk2 = torchattacks.PGD(model, eps=(8/255 * ratio), alpha=(2/255*ratio), steps=10*ratio, random_start=True)
    
    
    atk3 = torchattacks.CW(model, c=0.1, steps=1000, lr=0.01)
    atk4 = torchattacks.CW(model, c=1, steps=1000, lr=0.01)

    atk = torchattacks.MultiAttack([atk1, atk2])
    # atk = torchattacks.MultiAttack([atk1, atk2, atk3, atk4])
    atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    adv_images = atk(images, labels)
    
    return adv_images

def load_attack_model():
    # model = torchvision_models.resnet18(pretrained=True).to(device).eval()
    model = torchvision_models.inception_v3(pretrained=True).to(device).eval()
    return model

@torch.no_grad()
def get_attack_predict(model, images):
    pre = model(images.to(device))
    _, pre = torch.max(pre.data, 1)
    assert pre.shape[0] == images.shape[0]
    return pre
