import os
import torch
import time
from tqdm import tqdm

class Trainer:
    def __init__(self, model, data_loader, config, device):
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.device = device
        self.optimizer = torch.optim.SGD(params=[p for p in model.parameters() if p.requires_grad],
                                         lr=config['optimizer']['args']['lr'],
                                         momentum=config['optimizer']['args']['momentum'],
                                         weight_decay=config['optimizer']['args']['weight_decay'])
        self.num_epochs = config['trainer']['epochs']
        self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(config['class_weights'], dtype=torch.float32).to(device))

    def train(self):
        for epoch in range(self.num_epochs):
            start_time = time.time()
            self.model.train()
            epoch_loss = 0

            # tqdm을 사용하여 진행 상황 시각화
            data_loader_iter = tqdm(self.data_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}", dynamic_ncols=True)

            for imgs, targets_list in data_loader_iter:
                imgs = list(img.to(self.device) for img in imgs)
                targets = [{k: v.to(self.device) for k, v in target.items()} for targets in targets_list for target in targets]

                self.optimizer.zero_grad()
                loss_dict = self.model(imgs, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                self.optimizer.step()

                epoch_loss += losses.item()

                # tqdm 업데이트
                data_loader_iter.set_postfix(loss=epoch_loss)

            data_loader_iter.close()  # tqdm 종료

            print(f"Training Epoch {epoch + 1}/{self.num_epochs}, Loss: {epoch_loss}, Time: {time.time() - start_time}")

            # 모델 가중치 저장
            save_path = os.path.join(self.config['save_dir'], f'epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': epoch_loss,
            }, save_path)

            print(f"모델 가중치가 저장되었습니다: {save_path}")
