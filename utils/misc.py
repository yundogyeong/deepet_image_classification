import torch
import math
import json
import numpy as np
from typing import Union, Iterable
from datetime import datetime

def get_device(tag:str) -> None:
    if tag == "cpu":
        if torch.cuda.is_available():
            while True:
                flag = input("[INFO] cuda is available, switch to cuda? [y/n] : ")
                if flag == "y":
                    return torch.device("cuda")
                elif flag == "n":
                    return torch.device("cpu")
                else:
                    continue
        return torch.device("cpu")
    elif tag == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("[WARN] cuda is not available !!, return torch.device(\"cpu\")")
            return torch.device("cpu")

def save_dict_as_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def get_current_datetime() -> str:
    now = datetime.now()
    return now.strftime("%Y%m%d%H%M")[2:]

class AverageMeter:
    """
    평균값 및 표준편차를 계산하고 저장하는 클래스
    """
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """
        초기화 메서드
        """
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.var = 0
        self.std = 0

    def update(self, val: Union[torch.Tensor, int, float, Iterable], n: int = 1) -> None:
        """
        값을 입력받아 평균값과 표준편차를 계산하고 저장하는 메서드

        :param val: 입력값
        :param n: 입력값의 개수
        """
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu().item()

        if isinstance(val, Iterable):
            val = np.mean(val)

        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        if self.count > 1:
            self.var = ((self.count - 1) * self.var + (val - self.avg) ** 2) / self.count
            self.std = math.sqrt(self.var)