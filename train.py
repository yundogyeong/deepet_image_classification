import os
import argparse
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR
from tqdm import tqdm
from models import get_model
from utils import get_dataloader, get_current_datetime, AverageMeter, save_dict_as_json, get_device

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="../data/ImageNet_2012_rename")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--model", type=str, default="vgg", choices=["vgg", "vit"])
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--imgsz", type=int, default=224)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--optimizer", type=str, default='adam')
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--sch", type=str, default="step")
parser.add_argument("--step-size", type=int, default=10)
parser.add_argument("--gamma", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--save_interval", type=int, default=50)

# TODO LIST
# parser.add_argument("--amp")
# parser.add_argument("--ddp")

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

# dropout 비율 수정
def modify_dropout(model, p):
    for _, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.p = p
    return model

# optimizer 관련 코드
def get_optimizer(name:str):
    if name.lower() == 'sgd':
        return SGD
    elif name.lower() == 'adam':
        return Adam
    else:
        raise ValueError(name)
    
# scheduler 관련 코드
def get_scheduler(name:str):
    if name.lower() == "step":
        return StepLR
    elif name.lower() == 'exp':
        return ExponentialLR
    elif name.lower() == 'cosine':
        return CosineAnnealingLR
    else:
        raise ValueError(name)
    

if __name__ == "__main__":
    if not os.path.exists("./runs"):
        os.makedirs("./runs")
    
    opt = parser.parse_args()
    # 모델 가중치 저장 디렉토리
    SAVE_PATH, PTH_DIR = prepare(opt)
    
    DEVICE = get_device(opt.device)

    # 신경망
    net = modify_dropout(get_model(opt), opt.dropout).to(DEVICE)

    # 손실함수
    criterion = nn.CrossEntropyLoss()
    
    # 최적화
    optimizer_type = get_optimizer(opt.optimizer)
    optimizer = optimizer_type(net.parameters(), lr=opt.lr)
    
    # 스케줄러
    scheduler_type = get_scheduler(opt.sch)
    try:
        scheduler = scheduler_type(optimizer, step_size=opt.step_size, gamma=opt.gamma)
    except:
        scheduler = scheduler_type(optimizer, T_max=opt.step_size)    
    
    print(opt.root)
    # 데이터 파이프라인
    train_loader, val_loader = get_dataloader(opt.root, opt.batch_size, opt.imgsz, opt.num_workers)
    
    for e in range(1, opt.epochs+1):
        train_loss = AverageMeter()
        train_correct = 0
        train_total = 0
        
        # iteration check
        train_iteration = 0
        
        # 훈련모드로 설정(BN, Dropout 활성화)
        net.train()
        train_bar = tqdm(train_loader, ncols=120)
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            # forward
            preds = net(inputs)
            loss = criterion(preds, labels)
            
            # back-propagation
            loss.backward()
            optimizer.step()
            
            # compute accuracy during training
            _, predicted = torch.max(preds.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_loss.update(loss)
            
            # update iteration
            train_iteration += 1
            
            train_bar.set_description(
                f"# TRAIN[{e}/{opt.epochs}] loss = {train_loss.avg:.3f}, correct/total = {train_correct}/{train_total}"
            )