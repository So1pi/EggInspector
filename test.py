import torch
import os
from data_loader.egg_dataset import EggDataset
from model.model import get_model_instance_segmentation
from trainer.trainer import evaluate_model
from parse_config import parse_config


def main(config):
    # 검증 데이터셋 및 데이터로더 준비
    validation_dataset = EggDataset(config['transforms'], config['validation_data_path'])
    validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=config['batch_size'],
                                                         collate_fn=config['collate_fn'], shuffle=True)

    # 모델 초기화 및 장치 설정
    model = get_model_instance_segmentation(config['num_classes'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 모델 가중치 로드
    model_weights_path = config['model_weights_path']
    if os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path)['model_state_dict'])
    else:
        raise FileNotFoundError(f"모델 가중치 파일을 찾을 수 없습니다: {model_weights_path}")

    # 모델 평가
    all_predictions, all_targets = evaluate_model(model, validation_data_loader, device)

    # 평가 결과 출력
    len_all_targets = len(all_targets)
    len_all_predictions = len(all_predictions)
    print(f"Length of all_targets: {len_all_targets}")
    print(f"Length of all_predictions: {len_all_predictions}")


if __name__ == '__main__':
    config_file = 'config.json'
    config = parse_config(config_file)
    main(config)
