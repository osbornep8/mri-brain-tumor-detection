import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from datetime import datetime
from .model import TumorClassifier
from .dataset import BrainTumorDataset
from .transforms import calculate_normalization_values, create_transforms, basic_transform
from .utils import save_results
from .evaluate import evaluate_model
 
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30, device='cuda'):
    """
    Train the model with early stopping and learning rate scheduling
    Returns training history for plotting
    """
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'best_model': None,
        'best_epoch': 0,
        'best_val_loss': float('inf')
    }

    
    
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Early stopping check
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['best_model'] = model.state_dict().copy()
            history['best_epoch'] = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
            
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return history


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # Create output directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    weights_dir = os.path.join('models', 'weights', timestamp)
    results_dir = os.path.join('models', 'results', timestamp)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # First, create dataset with basic transform
    temp_dataset = BrainTumorDataset("brain_tumor_dataset", transform=basic_transform)
    temp_loader = DataLoader(temp_dataset, batch_size=32, shuffle=False)
    
    # Calculate normalization values
    mean, std = calculate_normalization_values(temp_loader)
    
    # Create proper transforms
    train_transform, test_transform = create_transforms(mean, std)
    
    # Create full dataset with proper transform
    full_dataset = BrainTumorDataset("brain_tumor_dataset", transform=train_transform)
    
    # Rest of your code remains the same...
    # First split: training vs test (90/10)
    total_size = len(full_dataset)
    test_size = int(0.1 * total_size)
    train_val_size = total_size - test_size
    
    train_val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_val_size, test_size]
    )
    
    # Second split: train vs validation (90/10 of training data)
    train_size = int(0.9 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_val_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize model
    model = TumorClassifier().to(device)
    
    # Calculate class weights by accessing the original dataset correctly
    original_dataset = train_dataset.dataset.dataset  # Access the original dataset through nested subsets
    train_indices = [train_dataset.indices[i] for i in range(len(train_dataset))]
    train_labels = [int(original_dataset.data[idx][1]) for idx in train_indices]
    
    num_pos = sum(train_labels)
    num_neg = len(train_labels) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos]).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5
        )
    
    
    # Train model
    history = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler, device=device
    )
    
    # Save model weights
    weight_path = os.path.join(weights_dir, 'Brain_Tumor_BClassifier_model.pth')
    torch.save(model.state_dict(), weight_path)
    
    # Evaluate model
    model.load_state_dict(history['best_model'])
    metrics = evaluate_model(model=model, test_loader=test_loader, device=device)
    
    # Save results
    save_results(history, metrics, results_dir)
    
    return model, history, metrics

if __name__ == '__main__':
    model, history, metrics = main()