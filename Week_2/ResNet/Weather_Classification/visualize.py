import matplotlib.pyplot as plt
import streamlit as st

def plot_figure(train_losses, val_losses):
    st.subheader("Training and Validation Loss")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(train_losses)
    ax[0].set_title('Training Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[1].plot(val_losses, color='orange')
    ax[1].set_title('Val Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    st.pyplot(fig)