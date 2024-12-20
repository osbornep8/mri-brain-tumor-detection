import torch
from torchvision import transforms

def calculate_normalization_values(loader):
    """Calculate dataset-specific mean and std values from a DataLoader"""
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        total_samples += batch_samples
    
    mean = mean / total_samples
    
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        std += ((images - mean.view(1, -1, 1))**2).mean(2).sum(0)
    
    std = torch.sqrt(std / total_samples)
    
    return mean.tolist(), std.tolist()

def create_transforms(mean, std):
    """Create train and test transforms with provided normalization values"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(43),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(
            degrees=0,
            shear=15,
            fill=0
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.02)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transform, test_transform

# Create basic transform for normalization calculation
basic_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])