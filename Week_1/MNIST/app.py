import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import streamlit as st
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
sys.path.append(str(Path(__file__).resolve().parents[2]))
from Week_1.MNIST.data_preprocessing import preprocess_data
from Week_1.MNIST.train_eval import train, evaluate  
from Week_1.MNIST.model import LeNetClassifier
from Week_1.MNIST.visualize import plot_result

@st.cache_resource
def load_model(model_path, num_classes=10, input_channels=1, input_size=5, weights_only=None):
    lenet_model = LeNetClassifier(num_classes, input_channels, input_size)
    lenet_model.load_state_dict(torch.load(f=model_path, map_location=torch.device('cpu'), weights_only=weights_only))
    lenet_model.eval()
    return lenet_model

def inference(image, model):
    img_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])
    img_new = img_transform(image).unsqueeze(0)
    with torch.no_grad():
        predictions = model(img_new)
    preds = nn.Softmax(dim=1)(predictions)
    p_max, yhat = torch.max(preds.data, 1)
    return p_max.item()*100, yhat.item()

def train_model():
    with st.spinner("Starting training..."):
        train_dataloader, valid_dataloader, test_dataloader, train_data = preprocess_data()
        num_classes = len(train_data.dataset.classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        lenet_model = LeNetClassifier(num_classes)
        lenet_model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(lenet_model.parameters(), lr=0.001, weight_decay=0)

        # Training and evaluation loop
        num_epochs = 10
        save_model = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(save_model, exist_ok=True)
        train_accs, train_losses = [], []
        eval_accs, eval_losses = [], []
        best_loss_eval = float('inf')

        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            
            # Training
            train_acc, train_loss = train(lenet_model, optimizer, criterion, train_dataloader, device, epoch)
            train_accs.append(train_acc)
            train_losses.append(train_loss)

            # Evaluation
            eval_acc, eval_loss = evaluate(lenet_model, criterion, valid_dataloader, device)
            eval_accs.append(eval_acc)
            eval_losses.append(eval_loss)

            # Save best model
            if eval_loss < best_loss_eval:
                model_save_path = os.path.join(save_model, "lenet_model.pt")
                torch.save(lenet_model.state_dict(), model_save_path)
                best_loss_eval = eval_loss

            # Print loss and accuracy for the epoch
            print("-" * 59)
            print(f"| End of epoch {epoch:3d} | Time: {time.time() - epoch_start_time:5.2f}s "
                f"| Train Accuracy {train_acc:8.3f} | Train Loss {train_loss:8.3f} "
                f"| Valid Accuracy {eval_acc:8.3f} | Valid Loss {eval_loss:8.3f}")
            print("-" * 59)
        
        test_acc, test_loss = evaluate(lenet_model, criterion, test_dataloader, device)
    st.success(f"Model trained and saved at {model_save_path}")
    return train_accs, train_losses, eval_accs, eval_losses, test_acc, test_loss

def crop_center(image):
    width, height = image.size
    new_edge = min(width, height)
    left = (width - new_edge) // 2
    top = (height - new_edge) // 2
    right = (width + new_edge) // 2
    bottom = (height + new_edge) // 2
    return image.crop((left, top, right, bottom))

def upload_image(model):
    if not st.session_state.model_trained:
        st.warning("You need to train the model first before predicting.")
    else:
        file = st.file_uploader("Please upload an image of a digit", type=["jpg", "png"])
        if file is not None:
            image = Image.open(file)
            image = crop_center(image)
            p, label = inference(image, model)
            st.image(image)
            st.success(f"The uploaded image is of the digit {label} with {p:.2f} % probability.") 
        
def run_sample(model):
    if not st.session_state.model_trained:
        st.warning("You need to train the model first before predicting.")
    else:
        example_image_path = os.path.join(os.path.dirname(__file__), 'demo_8.png')
        if os.path.exists(example_image_path):
            image = Image.open(example_image_path)
            p, label = inference(image, model)
            st.image(image)
            st.success(f"The image is of the digit {label} with {p:.2f} % probability.")
        else:
            st.error(f"Example image '{example_image_path}' not found.")  
                      
def main():
    st.title('Digit Recognition')

    with st.sidebar:
        st.header("🛠️ Configuration")
        st.subheader('Model: LeNet. Dataset: MNIST')
        
        button = st.button('Train Model')
        
        if not st.session_state.model_trained:
            st.warning("You need to train model first!")
        
        if button:
            train_accs, train_losses, eval_accs, eval_losses, test_accs, test_losses = train_model()
            st.session_state.model_trained = True
            st.session_state.train_accs = train_accs 
            st.session_state.train_losses = train_losses
            st.session_state.eval_accs = eval_accs
            st.session_state.eval_losses = eval_losses
            st.session_state.test_accs = test_accs
            st.session_state.test_losses = test_losses

    tab1, tab2 = st.tabs(["🔮 Predict", "📊 Model Performance"])
    with tab1:
        option = st.selectbox('How would you like to give the input?', ('Upload Image File', 'Run Example Image'))
                
        model_path = os.path.join(os.path.join(os.path.dirname(__file__), "models"), 'lenet_model.pt')
        
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            st.warning("No trained model found. Please train the model first.")
            return
        
        if option == "Upload Image File":
            upload_image(model)
        if option == "Run Example Image":
            run_sample(model)
            
    with tab2:
        if st.session_state.model_trained:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Accuracy", f"{st.session_state.train_accs[-1]:.4f}")
                st.metric("Validation Accuracy", f"{st.session_state.eval_accs[-1]:.4f}")
            with col2:
                st.metric("Test Accuracy", f"{st.session_state.test_accs:.4f}")
            plot_result(10, st.session_state.train_accs, st.session_state.eval_accs, st.session_state.train_losses, st.session_state.eval_losses) 
        else:
            st.warning("You need to train the model first before viewing performance.")


if __name__ == '__main__':
    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False 
    if "train_accs" not in st.session_state:
        st.session_state.train_accs = [] 
    if "train_losses" not in st.session_state:
        st.session_state.train_losses = [] 
    if "eval_accs" not in st.session_state:
        st.session_state.eval_accs = [] 
    if "eval_losses" not in st.session_state:
        st.session_state.eval_losses = [] 
    if "test_accs" not in st.session_state:
        st.session_state.test_accs = 0.0 
    if "test_losses" not in st.session_state:
        st.session_state.test_losses = 0.0 
    main()