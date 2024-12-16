import time
import torch

def train(model, optimizer, criterion, train_dataloader, device, epoch=0, log_interval=50, use_grad_clip=True):
    """
    Training function for one epoch.
    """
    model.train()
    total_acc, total_count = 0, 0
    losses = []
    start_time = time.time()

    for idx, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(inputs)

        # Compute loss
        loss = criterion(predictions, labels)
        losses.append(loss.item())

        # Backward and optimize
        loss.backward()
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)        
        optimizer.step()

        # Update accuracy
        total_acc += (predictions.argmax(1) == labels).sum().item()
        total_count += labels.size(0)

        if idx % log_interval == 0 and idx > 0:
            _ = time.time() - start_time
            print(f"| epoch {epoch:3d} | {idx:5d}/{len(train_dataloader):5d} batches "
                  f"| accuracy {total_acc / total_count:8.3f}")
            total_acc, total_count = 0, 0
            start_time = time.time()

    epoch_acc = total_acc / total_count
    epoch_loss = sum(losses) / len(losses)
    return epoch_acc, epoch_loss


def evaluate(model, criterion, valid_dataloader, device):
    """
    Evaluation function for one epoch.
    """
    model.eval()
    total_acc, total_count = 0, 0
    losses = []

    with torch.no_grad():
        for inputs, labels in valid_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            predictions = model(inputs)
            loss = criterion(predictions, labels)
            losses.append(loss.item())

            total_acc += (predictions.argmax(1) == labels).sum().item()
            total_count += labels.size(0)

    epoch_acc = total_acc / total_count
    epoch_loss = sum(losses) / len(losses)
    return epoch_acc, epoch_loss