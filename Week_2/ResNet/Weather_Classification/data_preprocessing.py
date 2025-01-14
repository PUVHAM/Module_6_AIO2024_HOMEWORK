import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class WeatherDataset(Dataset):
    def __init__(
        self,
        x, y,
        transform=None
    ):
        self.transform = transform
        self.img_paths = x
        self.labels = y

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]
    
def transform(img, img_size=(224, 224)):
    img = img.resize(img_size)
    img = np.array(img)[..., :3]
    img = torch.tensor(img).permute(2, 0, 1).float()
    normalized_img = img / 255.0

    return normalized_img

def create_weather_dataloader(x_train, y_train, x_test, y_test, x_val, y_val):
    train_dataset = WeatherDataset(
        x_train, y_train,
        transform=transform
    )
    val_dataset = WeatherDataset(
        x_val, y_val,
        transform=transform
    )
    test_dataset = WeatherDataset(
        x_test, y_test,
        transform=transform
    )
    
    train_batch_size = 512
    test_batch_size = 8

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0
    )
    return train_loader, val_loader, test_loader