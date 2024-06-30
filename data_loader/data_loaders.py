import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data_set.egg_dataset import EggDataset


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return tuple(zip(*batch)) if batch else list(batch)


class EggDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=(-45, 45), fill=(0,)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        ])
        val_size = int(validation_split * len(dataset))
        train_size = len(dataset) - val_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        super().__init__(self.train_dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, collate_fn=collate_fn)

        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                     collate_fn=collate_fn)

    def split_validation(self):
        return self.val_loader

    def __len__(self):
        return len(self.train_dataset)
