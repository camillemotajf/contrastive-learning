import torch
import torch.nn as nn
import torch.nn.functional as F
from src.encoder import TrafficEncoder

# ATENÇÃO: Importe o seu TrafficEncoder original aqui
# Exemplo: from src.models import TrafficEncoder 

class ContrastiveClusteringNet(nn.Module):
    # Adicionamos o 'encoder_out_dim=64' para bater com a saída do seu TrafficEncoder
    def __init__(self, input_dim, encoder_out_dim=64, num_clusters=2):
        super().__init__()
        # O seu encoder base
        self.encoder = TrafficEncoder(input_dim) 
        
        # Cabeça para o contraste de instâncias (Agora recebe o encoder_out_dim)
        self.instance_head = nn.Sequential(
            nn.Linear(encoder_out_dim, 32), # Entra 64, reduz para 32
            nn.ReLU(),
            nn.Linear(32, 16)               # Reduz para 16 (representação final de instância)
        )
        
        # Cabeça para o contraste de clusters (Agora recebe o encoder_out_dim)
        self.cluster_head = nn.Sequential(
            nn.Linear(encoder_out_dim, 32), # Entra 64, reduz para 32
            nn.ReLU(),
            nn.Linear(32, num_clusters),    # Sai 2 (probabilidades de cluster)
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        h = self.encoder(x)
        
        # Normalização L2 para a cabeça de instância
        z_instance = F.normalize(self.instance_head(h), dim=1)
        
        # Probabilidades de cluster
        p_cluster = self.cluster_head(h) 
        
        return z_instance, p_cluster
    
class InstanceLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z = torch.cat((z_i, z_j), dim=0) 
        
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        
        # TRUQUE DO PYTORCH: Zera a chance do modelo escolher a diagonal preenchendo com -inf
        sim_matrix.fill_diagonal_(-1e9)
        
        # Agora os índices positivos batem perfeitamente
        positives = torch.cat([torch.arange(batch_size, 2*batch_size), torch.arange(batch_size)]).to(z.device)
        
        return self.criterion(sim_matrix, positives)

class ClusterLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, p_i, p_j):
        p_i = p_i.T 
        p_j = p_j.T
        
        p_i = F.normalize(p_i, dim=1, p=2)
        p_j = F.normalize(p_j, dim=1, p=2)
        
        num_clusters = p_i.shape[0]
        p = torch.cat((p_i, p_j), dim=0)
        
        sim_matrix = torch.matmul(p, p.T) / self.temperature
        
        # TRUQUE DO PYTORCH
        sim_matrix.fill_diagonal_(-1e9)
        
        positives = torch.cat([torch.arange(num_clusters, 2*num_clusters), torch.arange(num_clusters)]).to(p.device)
        
        return self.criterion(sim_matrix, positives)