import torch
import torch.nn as nn
import torch.nn.functional as F

class TrafficEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=64):
        super(TrafficEncoder, self).__init__()
        # Arquitetura MLP densa (Pode ser trocada por LSTM para as 10 amostras passadas depois)
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, embed_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        
        # O PULO DO GATO DO CONTRASTIVE LEARNING: Normalização L2
        # Força todos os embeddings para a superfície de uma hiperesfera
        x = F.normalize(x, p=2, dim=1)
        return x