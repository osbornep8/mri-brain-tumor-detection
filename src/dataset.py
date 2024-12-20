import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from typing import List, Tuple
import numpy as np

class BrainTumorDataset(Dataset):
    """
    Dataset class to load the Brain MRI Images from the 'kaggle/input' folder.
    """
    def __init__(self, base_dir: str, transform=None):
        
        """
        Initialize the dataset with support for the varying file extensions of jpeg and a few png files.  
        
        Args:
            base_dir (str): Base directory path containing 'yes' and 'no' subdirectories
            transform (callable, optional): Transform to be applied to images
        """
        self.base_dir = Path(base_dir)
        self.transform = transform
        self.valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG')
        
        # Initialize lists to store paths and labels
        self.data: List[Tuple[str, int]] = []
        self.class_counts = {'yes': 0, 'no': 0}
        
        # Load and validate images
        self._load_dataset()
        
    def _load_dataset(self):
        """
        Load dataset while handling multiple image formats
        """
        # Load tumor images (label 1)
        tumor_dir = self.base_dir / 'yes'
        tumor_images = []
        for ext in self.valid_extensions:
            tumor_images.extend(list(tumor_dir.glob(f'*{ext}')))
        
        for img_path in tumor_images:
            try:
                with Image.open(img_path) as img:
                    img.verify()
                    self.data.append((str(img_path), 1))
                    self.class_counts['yes'] += 1
            except (IOError, SyntaxError) as e:
                    print(f"Warning: Skipping corrupted image {img_path}: {e}")
            
        # Load non-tumor images (label 0)
        no_tumor_dir = self.base_dir / 'no'
        no_tumor_images = []
        for ext in self.valid_extensions:
            no_tumor_images.extend(list(no_tumor_dir.glob(f'*{ext}')))
            
        for img_path in no_tumor_images:
            try:
                with Image.open(img_path) as img:
                    img.verify()
                    self.data.append((str(img_path), 0))
                    self.class_counts['no'] += 1
            except (IOError, SyntaxError) as e:
                    print(f"Warning: Skipping corrupted image {img_path}: {e}")
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        labels = int(label)
        
        if self.transform:
            image = self.transform(image)
    
        return image, labels
    


# src/dataset.py
