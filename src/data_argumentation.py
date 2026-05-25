import torch
import numpy as np
from torch.utils.data import Dataset

class AugmentedTrafficDataset(Dataset):
    def __init__(self, features, mask_prob=0.15, noise_std=0.05):
        """
        features: Matriz numpy de entrada (ex: 20000 x 68)
        mask_prob: Probabilidade de "esconder" uma feature (dropout tabular)
        noise_std: Desvio padrão do ruído gaussiano
        """
        # Normalização padrão
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) + 1e-8
        self.features = torch.tensor((features - mean) / std, dtype=torch.float32)
        
        self.mask_prob = mask_prob
        self.noise_std = noise_std

    def augment(self, x):
        # 1. Feature Masking (zera colunas aleatórias)
        mask = (torch.rand(x.shape) > self.mask_prob).float()
        x_masked = x * mask
        
        # 2. Gaussian Noise
        noise = torch.randn(x.shape) * self.noise_std
        
        return x_masked + noise

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        
        # Gera duas VISÕES diferentes da mesma amostra
        view_a = self.augment(x)
        view_b = self.augment(x)
        
        return view_a, view_b