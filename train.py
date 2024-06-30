import torch
import os

from data_loader.data_loaders import EggDataLoader
from data_set.egg_dataset import EggDataset
from model.model import get_model_instance_segmentation
from trainer.trainer import Trainer
from parse_config import parse_config


def main(config):
    # 데이터셋 및 데이터로더 준비
    dataset = EggDataset(data_transforms=config['data_transforms']['train'], data_path=config['data_path'])
    data_loader = EggDataLoader(dataset=dataset, batch_size=config['batch_size'], shuffle=True,
                                validation_split=config['data_loader']['args']["validation_split"],
                                num_workers=config['data_loader']['args']["num_workers"])

    # 모델 초기화 및 장치 설정
    model = get_model_instance_segmentation(config['num_classes'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 트레이너 초기화 및 학습 시작
    trainer = Trainer(model, data_loader, config, device)
    trainer.train()


if __name__ == '__main__':
    config_file = 'config.json'
    config = parse_config(config_file)
    main(config)
