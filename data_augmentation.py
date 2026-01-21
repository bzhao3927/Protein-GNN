# data_augmentation.py
import numpy as np
from scipy.spatial.transform import Rotation

def augment_protein_coords(coords, augment_prob=0.5):
    """Random rotation + small noise"""
    if np.random.rand() > augment_prob:
        return coords
    
    # Random rotation
    R = Rotation.random().as_matrix()
    coords_rot = coords @ R.T
    
    # Small Gaussian noise (0.1 Å standard deviation)
    noise = np.random.randn(*coords.shape) * 0.1
    coords_aug = coords_rot + noise
    
    return coords_aug