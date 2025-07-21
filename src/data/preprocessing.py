# needs to be changed according to the data and requirements

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from scipy import ndimage
import cv2

class XRayPreprocessor:
    """X-ray scattering specific preprocessing pipeline"""
    
    def __init__(self, image_size=224):
        self.image_size = image_size
        
    def beam_stop_mask(self, image, center=None, radius=None):
        """Apply beam stop masking"""
        if center is None:
            center = (image.shape[0]//2, image.shape[1]//2)
        if radius is None:
            radius = min(image.shape[:2]) // 20
            
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
        image[mask] = 0
        return image
    
    def dead_pixel_correction(self, image, threshold=3):
        """Simple dead pixel correction using median filtering"""
        median_filtered = ndimage.median_filter(image, size=3)
        diff = np.abs(image - median_filtered)
        dead_pixels = diff > threshold * np.std(diff)
        image[dead_pixels] = median_filtered[dead_pixels]
        return image
    
    def log_intensity_transform(self, image):
        """Apply logarithmic intensity transformation"""
        # Add small constant to avoid log(0)
        return np.log(image + 1)
    
    def radial_normalization(self, image):
        """Simple radial normalization"""
        center = (image.shape[0]//2, image.shape[1]//2)
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        # Normalize by radial average
        r_int = r.astype(int)
        max_r = min(center)
        for radius in range(1, max_r):
            mask = (r_int == radius)
            if np.any(mask):
                mean_val = np.mean(image[mask])
                if mean_val > 0:
                    image[mask] /= mean_val
        return image
    
    def preprocess(self, image):
        """Complete preprocessing pipeline"""
        # Convert to float32
        image = image.astype(np.float32)
        
        # X-ray specific preprocessing
        image = self.dead_pixel_correction(image)
        image = self.beam_stop_mask(image)
        image = self.log_intensity_transform(image)
        image = self.radial_normalization(image)
        
        # Standard preprocessing
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Convert to 3 channels (DINOv2 expects RGB)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=0)
        
        return torch.FloatTensor(image)

class XRayDataset(torch.utils.data.Dataset):
    """Dataset for X-ray images"""
    
    def __init__(self, image_paths, labels=None, preprocessor=None, augment=True):
        self.image_paths = image_paths
        self.labels = labels
        self.preprocessor = preprocessor or XRayPreprocessor()
        self.augment = augment
        
        if augment:
            self.transforms = transforms.Compose([
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image (assuming .tif or .npy format)
        img_path = self.image_paths[idx]
        if img_path.endswith('.npy'):
            image = np.load(img_path)
        else:
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        # Preprocess
        image = self.preprocessor.preprocess(image)
        
        # Apply augmentations
        if self.augment and self.training:
            image = self.transforms(image)
        
        if self.labels is not None:
            return image, torch.FloatTensor(self.labels[idx])
        return image
