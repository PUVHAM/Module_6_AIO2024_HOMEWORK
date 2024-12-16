import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import streamlit as st
from pathlib import Path
from torchtext.data.utils import get_tokenizer
from data_preprocessing import preprocess_text, create_ntc_dataloader, load_vocab
from model import TextCNN
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[2]))

from Week_1.MNIST.train_eval import train, evaluate
from Week_1.MNIST.visualize import plot_result

tokenizer = get_tokenizer("basic_english")
idx2label = {0: 'negative', 1:'positive'}

def load_model(model_path, vocab_size=10000, embedding_dim=100, num_classes=2):
    model = TextCNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        kernel_sizes=[3, 4, 5],
        num_filters=100,
        num_classes=num_classes
    )
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
    model.eval()
    return model

def inference(sentence, vocabulary, model):
    sentence = preprocess_text(sentence)
    encoded_sentence = vocabulary(tokenizer(sentence))
    encoded_sentence = torch.tensor(encoded_sentence)
    padding = 5 - len(encoded_sentence) if len(encoded_sentence) < 5 else 0
    if padding > 0:
        encoded_sentence = F.pad(encoded_sentence, (0, padding), value=0)
    encoded_sentence = torch.unsqueeze(encoded_sentence, 1)

    with torch.no_grad():
        predictions = model(encoded_sentence)
    preds = nn.Softmax(dim=1)(predictions)
    p_max, yhat = torch.max(preds.data, 1)
    return round(p_max.item(), 2)*100, yhat.item()


def train_model():
    with st.spinner("Starting training..."):
        train_dataloader, valid_dataloader, test_dataloader = create_ntc_dataloader()
        num_classes = 2
        vocabulary = load_vocab()
        vocab_size = len(vocabulary)
        embedding_dim = 100
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TextCNN(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            kernel_sizes=[3, 4, 5],
            num_filters=100,
            num_classes=num_classes
        )
        model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

        # Training and evaluation loop
        num_epochs = 10
        save_model = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(save_model, exist_ok=True)
        train_accs, train_losses = [], []
        eval_accs, eval_losses = [], []
        best_loss_eval = 100

        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            
            # Training
            train_acc, train_loss = train(model, optimizer, criterion, train_dataloader, device, epoch, use_grad_clip=False)
            train_accs.append(train_acc)
            train_losses.append(train_loss)

            # Evaluation
            eval_acc, eval_loss = evaluate(model, criterion, valid_dataloader, device)
            eval_accs.append(eval_acc)
            eval_losses.append(eval_loss)

            # Save best model
            if eval_loss < best_loss_eval:
                model_save_path = os.path.join(save_model, "text_cnn.pt")
                torch.save(model.state_dict(), model_save_path)
                best_loss_eval = eval_loss

            # Print loss and accuracy for the epoch
            print("-" * 59)
            print(f"| End of epoch {epoch:3d} | Time: {time.time() - epoch_start_time:5.2f}s "
                f"| Train Accuracy {train_acc:8.3f} | Train Loss {train_loss:8.3f} "
                f"| Valid Accuracy {eval_acc:8.3f} | Valid Loss {eval_loss:8.3f}")
            print("-" * 59)
        
        test_acc, test_loss = evaluate(model, criterion, test_dataloader, device)
    st.success(f"Model trained and saved at {model_save_path}")
    return train_accs, train_losses, eval_accs, eval_losses, test_acc, test_loss

def input_text(model):
    if not st.session_state.model_trained:
        st.warning("You need to train the model first before predicting.")
    else:
        text_input = st.text_input("Sentence: ", "Đồ ăn ở quán này quá tệ luôn!")
        vocabulary = load_vocab()
        p, idx = inference(text_input, vocabulary, model)
        label = idx2label[idx]
        st.success(f'Sentiment: {label} with {p:.2f} % probability.')  
                      
def main():
    st.title('Sentiment Analysis')

    with st.sidebar:
        st.header("🛠️ Configuration")
        st.subheader('Model: Text CNN. Dataset: NTC-SCV')
        
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
                
        model_path = os.path.join(os.path.join(os.path.dirname(__file__), "models"), 'text_cnn.pt')
        
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            st.warning("No trained model found. Please train the model first.")
            return
             
        input_text(model)
            
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