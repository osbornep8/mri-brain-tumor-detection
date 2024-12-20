import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

def plot_training_history(history):
    """
    Plot training and validation metrics
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Accuracy Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(conf_matrix):
    """
    Plot confusion matrix heatmap
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt='d',
        cmap='Blues',
        xticklabels=['No Tumor', 'Tumor'],
        yticklabels=['No Tumor', 'Tumor']
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt.gcf()

def save_results(history, metrics, save_dir):
    """
    Save training history, metrics, and plots
    """
    # Save metrics to JSON
    metrics_file = os.path.join(save_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump({
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
            'auc_roc': float(metrics['auc_roc'])
        }, f, indent=4)
    
    # Save training history plot
    plt.figure()
    plot_training_history(history)
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()
    
    # Save confusion matrix plot
    plt.figure()
    plot_confusion_matrix(metrics['conf_matrix'])
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Save metrics in text format
    with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
        f.write("Test Set Metrics:\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"ROC AUC Score: {metrics['auc_roc']:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write("                 Predicted No Tumor  Predicted Tumor\n")
        f.write(f"Actual No Tumor      {metrics['conf_matrix'][0][0]}                {metrics['conf_matrix'][0][1]}\n")
        f.write(f"Actual Tumor         {metrics['conf_matrix'][1][0]}                {metrics['conf_matrix'][1][1]}\n")