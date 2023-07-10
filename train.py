import os
import argparse
import shutil
import torch
import torch.nn as nn
from typing import Union
from torch.optim import SGD, Adam
from tqdm import tqdm
from models import get_model
from utils import get_dataloader, get_current_datetime, AverageMeter, save_dict_as_json, get_device

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="../data/ImageNet_2012_rename")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--model", type=str, default="vgg", choices=["vgg", "vit"])
parser.add_argument("--imgsz", type=int, default=224)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--optimizer", type=str, default='sgd')
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--step", type=int, default=10)
parser.add_argument("--gamma", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--save_interval", type=int, default=50)

def prepare(opt):
    SAVE_PATH = os.path.join("./runs", opt.model.upper()+"_"+get_current_datetime())
    if os.path.exists(SAVE_PATH):
        shutil.rmtree(SAVE_PATH)
        os.makedirs(SAVE_PATH)
    else:
        # os.path.exists(SAVE_PATH) == False
        os.makedirs(SAVE_PATH)
    TRAIN_CONFIG = os.path.join(SAVE_PATH, "config.json")
    save_dict_as_json(TRAIN_CONFIG, vars(opt))
    
    PTH_DIR = os.path.join(SAVE_PATH, "weights")
    os.makedirs(PTH_DIR)
    
    return SAVE_PATH, PTH_DIR

# optimizer 관련 코드
def get_optimizer(name:str):
    if name.lower() == 'sgd':
        return SGD
    elif name.lower() == 'adam':
        return Adam

if __name__ == "__main__":
    if not os.path.exists("./runs"):
        os.makedirs("./runs")
    
    opt = parser.parse_args()
    # 모델 가중치 저장 디렉토리
    SAVE_PATH, PTH_DIR = prepare(opt)
    
    DEVICE = get_device(opt.device)

    # 신경망
    net = get_model(opt).to(DEVICE)
    
    # 손실함수
    criterion = nn.CrossEntropyLoss()
    
    # 최적화
    optimizer_type = get_optimizer(opt.optimizer)
    optimizer = optimizer_type(net.parameters(), lr=opt.lr)
    
    # 데이터 파이프라인
    train_loader, val_loader = get_dataloader(opt.root, opt.batch_size, opt.imgsz, opt.num_workers)
    
    for e in range(1, opt.epochs+1):
        pass