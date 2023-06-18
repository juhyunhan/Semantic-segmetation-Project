#main.py에 데이터셋을 가져와서 썼었는데,,
#사람 맘이지만 이걸 만들어서 녹여 보겟읍니다
#collate 평션이나 build 이런거 있으니꼐

from torchvision import transforms
from .CT_dataset import CT_dataset
from torch.utils.data import DataLoader
import torch


def build_transforms():
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transformer

def build_dataloader(data_dir, batch_size=4):
    transform = build_transforms()
    
    dataloaders = {}
    train_dataset = CT_dataset(data_dir=data_dir, phase="train", transforms=transform)
    dataloaders["train"] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    val_dataset = CT_dataset(data_dir=data_dir, phase="val", transforms=transform)
    dataloaders["val"] = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return dataloaders


def collate_fn(batch):
    images = []
    targets = []
    for a, b in batch: 
        images.append(a)
        targets.append(b)
    images = torch.stack(images, dim=0) 
    targets = torch.stack(targets, dim=0)

    return images, targets