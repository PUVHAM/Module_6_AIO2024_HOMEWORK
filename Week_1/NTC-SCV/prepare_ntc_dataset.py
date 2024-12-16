import os
import zipfile
import requests
import pandas as pd

class DatasetDownloadError(Exception):
    """Custom exception for dataset download errors."""
    pass

def download_and_prepare_ntc_dataset():
    _ = "https://github.com/congnghia0609/ntc-scv" # repo dataset url
    data_test_url = "https://github.com/congnghia0609/ntc-scv/raw/master/data/data_test.zip"
    data_train_url = "https://github.com/congnghia0609/ntc-scv/raw/master/data/data_train.zip"

    base_path = os.path.dirname(__file__)
    data_test_zip = os.path.join(base_path, "data_test.zip")
    data_train_zip = os.path.join(base_path, "data_train.zip")

    _download_file(data_test_url, data_test_zip)
    _download_file(data_train_url, data_train_zip)

    data_dir = os.path.join(base_path, "data")
    os.makedirs(data_dir, exist_ok=True)

    _unzip_file(data_test_zip, data_dir)
    _unzip_file(data_train_zip, data_dir)

    os.remove(data_test_zip)
    os.remove(data_train_zip)

    print("NTC dataset downloaded and prepared successfully.")

def _download_file(url, output_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded file from {url} to {output_path}")
    else:
        raise DatasetDownloadError(f"Failed to download file from {url}. HTTP Status Code: {response.status_code}")
    
def load_data_from_path(folder_path):
    examples = []
    for label in os.listdir(folder_path):
        full_path = os.path.join(folder_path, label)
        for file_name in os.listdir(full_path):
            file_path = os.path.join(full_path, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            sentence = " ".join(lines)
            if label == "neg":
                label = 0
            if label == "pos":
                label = 1
            data = {
                'sentence': sentence,
                'label': label
            }
            examples.append(data)
    return pd.DataFrame(examples)

def load_raw_dataset():
    base_path = os.path.dirname(__file__)
    pickle_path = os.path.join(base_path, "raw_dataset.pkl")

    if os.path.exists(pickle_path):
        print("Loading dataset from cached pickle file.")
        return pd.read_pickle(pickle_path)

    if not os.path.exists(os.path.join(base_path, 'data')):
        download_and_prepare_ntc_dataset()

    folder_paths = {
        'train': os.path.join(base_path, 'data', 'data_train', 'train'),
        'valid': os.path.join(base_path, 'data', 'data_train', 'test'),
        'test': os.path.join(base_path, 'data', 'data_test', 'test')
    }

    train_df = load_data_from_path(folder_paths['train'])
    valid_df = load_data_from_path(folder_paths['valid'])
    test_df = load_data_from_path(folder_paths['test'])

    dataset = {'train': train_df, 'valid': valid_df, 'test': test_df}
    pd.to_pickle(dataset, pickle_path)
    print(f"Dataset cached to {pickle_path}")
    return dataset

def _unzip_file(zip_path, extract_to):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"The zip file '{zip_path}' does not exist.")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted '{zip_path}' to '{extract_to}'")