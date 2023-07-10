from .vgg import VGG
# from .resnet import RESNET18, RESNET50
from .vit import VIT

def get_model(opt):
    return eval(opt.model.upper())()