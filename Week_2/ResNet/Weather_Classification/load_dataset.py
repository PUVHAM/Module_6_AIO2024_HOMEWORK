import os
import gdown
import zipfile
from sklearn.model_selection import train_test_split

def download_weather_dataset(folder_id="1R9R8HVG1sEWkeMI-5LazLMC97kZ_hLDP"):
    base_path = os.path.dirname(__file__)
    root_data_dir = os.path.join(base_path, "weather-dataset", "dataset")
    zip_path = os.path.join(base_path, "weather_dataset.zip")
    
    os.makedirs(root_data_dir, exist_ok=True)
    url = f"https://drive.google.com/uc?id={folder_id}"
    
    try:
        print("Downloading dataset...")
        gdown.download(url, zip_path, quiet=False)
        
        print("Extracting dataset...")
        _unzip_file(zip_path, base_path)
        
        os.remove(zip_path)
        
    except Exception as e:
        print(f"Error occurred during download: {str(e)}")
        return None

def prepare_weather_data(root_data_dir):
    try:
        classes = {
            label_idx: class_name 
            for label_idx, class_name in enumerate(
                sorted(os.listdir(root_data_dir))
            )
        }
        
        img_paths = []
        labels = []
        for label_idx, class_name in classes.items():
            class_dir = os.path.join(root_data_dir, class_name)
            for img_filename in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_filename)
                img_paths.append(img_path)
                labels.append(label_idx)
                
        return img_paths, labels, classes
        
    except Exception as e:
        print(f"Error occurred during data preparation: {str(e)}")
        return None, None, None

def split_dataset(img_paths, labels, seed):
    val_size = 0.2
    test_size = 0.125
    is_shuffle = True

    x_train, x_val, y_train, y_val = train_test_split(
        img_paths, labels,
        test_size=val_size,
        random_state=seed,
        shuffle=is_shuffle
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x_train, y_train,
        test_size=test_size,
        random_state=seed,
        shuffle=is_shuffle
    )
    return x_train, y_train, x_test, y_test, x_val, y_val

def _unzip_file(zip_path, extract_to):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"The zip file '{zip_path}' does not exist.")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted '{zip_path}' to '{extract_to}'")
    
if __name__ == "__main__":
    download_weather_dataset()