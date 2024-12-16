import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_mnist_data(root=os.path.join(os.path.dirname(__file__), 'data'), valid_ratio=0.9):
    # Load the raw MNIST dataset
    train_data = datasets.MNIST(root=root, train=True, download=True)
    test_data = datasets.MNIST(root=root, train=False, download=True)

    # Split training data into training and validation
    n_train_examples = int(len(train_data) * valid_ratio)
    n_valid_examples = len(train_data) - n_train_examples

    train_data, valid_data = random_split(
        train_data, [n_train_examples, n_valid_examples]
    )

    return train_data, valid_data, test_data

def compute_mean_std(dataset):
    mean = dataset.dataset.data.float().mean() / 255
    std = dataset.dataset.data.float().std() / 255
    return mean, std

def get_transforms(mean, std):
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])

    return train_transforms, test_transforms

def apply_transforms(train_data, valid_data, train_transforms, test_transforms, test_data):
    train_data.dataset.transform = train_transforms
    valid_data.dataset.transform = test_transforms
    test_data.transform = test_transforms

def create_dataloaders(train_data, valid_data, test_data, batch_size=256):
    train_dataloader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=batch_size,
        num_workers=0
    )

    valid_dataloader = DataLoader(
        valid_data,
        batch_size=batch_size,
        num_workers=0
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=0
    )

    return train_dataloader, valid_dataloader, test_dataloader

def preprocess_data():
    train_data, valid_data, test_data = load_mnist_data()
    mean, std = compute_mean_std(train_data)
    train_transforms, test_transforms = get_transforms(mean, std)
    apply_transforms(train_data, valid_data, train_transforms, test_transforms, test_data)
    train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(train_data, valid_data, test_data)
    return train_dataloader, valid_dataloader, test_dataloader, train_data