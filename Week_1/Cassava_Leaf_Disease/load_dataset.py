import os
import zipfile
import requests
from PIL import Image
from torchvision import datasets, transforms

class DatasetDownloadError(Exception):
    """Custom exception for dataset download errors."""
    pass

def download_dataset():
    url = "https://storage.googleapis.com/emcassavadata/cassavaleafdata.zip"
    output_path = os.path.join(os.path.dirname(__file__), "cassavaleafdata.zip")

    # Download the dataset using requests
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded dataset to {output_path}")
    else:
        raise DatasetDownloadError(f"Failed to download dataset. HTTP Status Code: {response.status_code}")

    extract_to = os.path.dirname(__file__)
    _unzip_file(output_path, extract_to)
    os.remove(output_path)

def _unzip_file(zip_path, extract_to):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"The zip file '{zip_path}' does not exist.")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted '{zip_path}' to '{extract_to}'")
    
def loader(path):
    return Image.open(path)

def create_image_folder():
    
    data_paths = {
        'train': os.path.join(os.path.dirname(__file__), 'cassavaleafdata' , 'train'),
        'valid': os.path.join(os.path.dirname(__file__), 'cassavaleafdata' , 'validation'),
        'test': os.path.join(os.path.dirname(__file__), 'cassavaleafdata' , 'test')
    }
    
    train_transforms = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
    ])
    
    train_data = datasets.ImageFolder(
        root=data_paths['train'],
        loader=loader,
        transform=train_transforms
    )
    
    valid_data = datasets.ImageFolder(
        root=data_paths['valid'], 
        transform=train_transforms
    )
    
    test_data = datasets.ImageFolder(
        root=data_paths['test'],
        transform=train_transforms
    )
    return train_data, valid_data, test_data

if __name__ == "__main__":
    download_dataset()