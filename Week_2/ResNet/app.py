import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import streamlit as st
import numpy as np
from PIL import Image
from config import AppConfig
from Weather_Classification.visualize import plot_figure
from Weather_Classification.train_eval import fit, evaluate
from Weather_Classification.model import ResNet, ResidualBlock
from Weather_Classification.load_dataset import prepare_weather_data, split_dataset, download_weather_dataset
from Weather_Classification.data_preprocessing import create_weather_dataloader, transform

def set_seed(seed=59):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(lr=1e-2, epochs=15):
    seed = set_seed()
    with st.spinner("Starting training..."):
        root_data_dir = os.path.join(os.path.dirname(__file__), "Weather_Classification", "weather-dataset")
        if not os.path.exists(root_data_dir):
            download_weather_dataset()
        img_paths, labels, classes = prepare_weather_data(os.path.join(root_data_dir, "dataset"))
        x_train, y_train, x_test, y_test, x_val, y_val = split_dataset(img_paths, labels, seed)
        train_loader, val_loader, test_loader = create_weather_dataloader(x_train, y_train, x_test, y_test, x_val, y_val)
        
        n_classes = len(list(classes.keys()))
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = ResNet(
            residual_block=ResidualBlock,
            n_blocks_lst=[2, 2, 2, 2],
            n_classes=n_classes
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=0,
            momentum=0
        )

        train_losses, val_losses = fit(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            epochs
        )
        
        _, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )
        _, test_acc = evaluate(
            model,
            test_loader,
            criterion,
            device
        )
    return train_losses, val_acc, val_losses, test_acc  

def predict(image, model, class_names, device="cpu"):
    """
    Perform inference on the input image and return the predicted class.
    """
    image_tensor = transform(image).to(device)
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_class = class_names[predicted_idx.item()]
    return predicted_class

def handle_prediction(config, uploaded_file):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Predict'):
            model_path = os.path.join(os.path.dirname(__file__), "Weather_Classification", 'best_model.pth')
            if os.path.exists(model_path):
                model = ResNet(
                    residual_block=ResidualBlock,
                    n_blocks_lst=[2, 2, 2, 2],
                    n_classes=len(config.get_weather_classes())
                ).to("cpu")
                checkpoint = torch.load(model_path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                with st.spinner('Predicting...'):
                    prediction = predict(image, model, config.get_weather_classes())
                    st.success(f'Predicted Weather: {prediction}')
            else:
                st.error("No trained model found. Please train the model first.")
                      
def main():
    config = AppConfig()
    
    st.title('Weather Image Classification with ResNet')

    with st.sidebar:
        st.header("🛠️ Model Configuration")
        
        learning_rate = st.slider('Learning Rate', 1e-4, 1e-1, 1e-2, format='%.4f')
        epochs = st.slider('Number of Epochs', 5, 50, 15)
        
        train_button = st.button('Train Model')
        
        if not st.session_state.model_trained:
            st.warning("Model needs to be trained first!")


    tab1, tab2 = st.tabs(["🔮 Predict", "📊 Training Metrics"])
    
    with tab1:
        st.header("Weather Classification")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        handle_prediction(config, uploaded_file)
        
    with tab2:
        if st.session_state.model_trained:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Validation Accuracy", f"{st.session_state.val_acc:.4f}")
            with col2:
                st.metric("Test Accuracy", f"{st.session_state.test_acc:.4f}")
            
            # Loss plots
            plot_figure(st.session_state.train_losses, st.session_state.val_losses)
        else:
            st.info("Training metrics will appear here after training and predicting.")
            
    # Training process
    if train_button:
        with st.spinner('Training in progress...'):
            model_path = os.path.join(os.path.dirname(__file__), "Weather_Classification", "best_model.pth")
            if not os.path.exists(model_path):
                train_losses, val_acc, val_losses, test_acc = train_model(
                    lr=learning_rate,
                    epochs=epochs
                )
                # Update session state
                st.session_state.model_trained = True
                st.session_state.train_losses = train_losses
                st.session_state.val_losses = val_losses
                st.session_state.val_acc = val_acc
                st.session_state.test_acc = test_acc
            else:
                checkpoint = torch.load(model_path)
                st.session_state.model_trained = True
                st.session_state.train_losses = checkpoint["train_losses"]
                st.session_state.val_losses = checkpoint["val_losses"]
                
                root_data_dir = os.path.join(os.path.dirname(__file__), "Weather_Classification", "weather-dataset")
                if not os.path.exists(root_data_dir):
                    download_weather_dataset()
                img_paths, labels, classes = prepare_weather_data(os.path.join(root_data_dir, "dataset"))
                x_train, y_train, x_test, y_test, x_val, y_val = split_dataset(img_paths, labels, set_seed())
                _, val_loader, test_loader = create_weather_dataloader(x_train, y_train, x_test, y_test, x_val, y_val)
                
                n_classes = len(list(classes.keys()))
                device = "cpu"

                model = ResNet(
                    residual_block=ResidualBlock,
                    n_blocks_lst=[2, 2, 2, 2],
                    n_classes=n_classes
                ).to(device)
                
                criterion = nn.CrossEntropyLoss()
                
                _, val_acc = evaluate(
                    model,
                    val_loader,
                    criterion,
                    device
                )
                _, test_acc = evaluate(
                    model,
                    test_loader,
                    criterion,
                    device
                )
                st.session_state.val_acc = val_acc
                st.session_state.test_acc = test_acc
            
        st.success('Training completed!')
        st.rerun()
    
 
if __name__ == '__main__':
    # Initialize session state variables
    session_states = {
        "model_trained": False,
        "train_losses": [],
        "val_losses": [],
        "val_acc": 0.0,
        "test_acc": 0.0
    }
    
    for key, value in session_states.items():
        if key not in st.session_state:
            st.session_state[key] = value
            
    main()