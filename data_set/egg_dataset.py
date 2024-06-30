import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import warnings
from bs4 import BeautifulSoup


class EggDataset(Dataset):
    def __init__(self, data_transforms, data_path):
        self.transform = T.Compose([
            T.ToTensor(),
            T.RandomRotation(degrees=(-45, 45), fill=(0,)),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        ])  # 데이터 변환을 self.transform으로 설정
        self.data_path = data_path
        self.imgs = sorted(os.listdir(os.path.join(self.data_path, '01.원천데이터', 'TS_02. COLOR')))
        self.num_classes = 6

    def __getitem__(self, idx):
        file_image = self.imgs[idx]
        file_label = self.imgs[idx][:-3] + 'xml'
        img_path = os.path.join(self.data_path, '01.원천데이터', 'TS_02. COLOR', file_image)
        label_path = os.path.join(self.data_path, '02.라벨링데이터', 'TL_02. COLOR', file_label)

        img = Image.open(img_path).convert("RGB")
        target = self.generate_target(label_path)

        if target is None or 'boxes' not in target or 'labels' not in target or target['boxes'].numel() == 0:
            print(f"인덱스 {idx}는 모든 주석이 무시되었으므로 건너뜁니다.")
            return None, None  # 이미지와 타겟 모두 None으로 반환

        img = self.transform(img)  # self.transform을 사용하여 데이터 변환 적용

        target_boxes = target["boxes"]
        target_labels = target["labels"]

        target = [{"boxes": target_boxes, "labels": target_labels, "image_id": torch.tensor([idx])}]
        return img, target

    def __len__(self):
        return len(self.imgs)

    def generate_box(self, obj):
        xmin = float(obj.find('x_min').text)
        ymin = float(obj.find('y_min').text)
        xmax = float(obj.find('x_max').text)
        ymax = float(obj.find('y_max').text)
        return [xmin, ymin, xmax, ymax]

    def generate_label(self, obj):
        adjust_label = 1
        default_label = 0
        if obj.find('state').text == "1":
            return 0 + adjust_label
        elif obj.find('state').text == "2":
            return 1 + adjust_label
        elif obj.find('state').text == "3":
            return 2 + adjust_label
        elif obj.find('state').text == "4":
            return 3 + adjust_label
        elif obj.find('state').text == "5":
            return 4 + adjust_label
        return default_label  # 기본 라벨 값 반환

    def generate_target(self, file):
        with open(file) as f:
            data = f.read()
            soup = BeautifulSoup(data, "xml")
            objects = soup.find_all("bndbox")

            num_objs = len(objects)

            if num_objs == 0:
                print(f"{file}에 유효한 주석이 없습니다.")
                # 주석이 없는 경우 빈 바운딩 박스와 기본 라벨 반환
                boxes = torch.as_tensor([], dtype=torch.float32)
                labels = torch.as_tensor([self.num_classes], dtype=torch.int64)
                target = {"boxes": boxes, "labels": labels}
                return target

            boxes = []
            labels = []

            for i in objects:
                box = self.generate_box(i)
                label = self.generate_label(i)

                # 유효한 바운딩 박스만 추가
                if box is not None:
                    boxes.append(box)
                    labels.append(label)

            if not boxes or not labels:
                print(f"{file}에 유효한 주석이 없습니다.")
                # 유효한 주석이 없는 경우 빈 바운딩 박스와 기본 라벨 반환
                boxes = torch.as_tensor([], dtype=torch.float32)
                labels = torch.as_tensor([self.num_classes], dtype=torch.int64)

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            target = {"boxes": boxes, "labels": labels}
            return target
