import json
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def process_traffic_data(file_unsafe, file_bots):
    """
    Lê os arquivos JSON, extrai features manuais e aplica 
    N-Grams de Caracteres na string bruta dos headers HTTP.
    """
    raw_headers = []
    manual_features = []
    labels = []
    
    # Função interna para processar cada arquivo mantendo a padronização
    def load_data(file_path, label_value):
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        for item in data:
            # A string completa captura erros de sintaxe e ordem anômala de chaves
            headers_str = item.get('headers', '{}')
            request_str = item.get('request', '{}')
            
            raw_headers.append(headers_str)
            
            # Features numéricas de apoio
            feat_vector = [
                len(headers_str),          # Tamanho da string do header
                len(request_str),          # Tamanho da string do request
                headers_str.count(':'),    # Estimativa da quantidade de chaves
                request_str.count(':')
            ]
            manual_features.append(feat_vector)
            labels.append(label_value)

    # Label 0: unsafe (tráfego de usuários reais)
    # Label 1: bots (tráfego automatizado)
    load_data(file_unsafe, 0)
    load_data(file_bots, 1)
    
    manual_features = np.array(manual_features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    # --- PIPELINE N-GRAMS DE CARACTERES ---
    # Transforma o texto em frequências de sequências de 2 a 4 caracteres
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(raw_headers)
    
    # Reduz as 5000 dimensões esparsas para 64 dimensões densas
    svd = TruncatedSVD(n_components=64, random_state=42)
    header_embeddings = svd.fit_transform(tfidf_matrix)
    
    # Concatena os embeddings textuais (64 dim) com as features numéricas (4 dim)
    # Resultado final: Cada requisição individual torna-se um vetor contínuo de 68 dimensões.
    final_features = np.hstack((header_embeddings, manual_features))
    
    return final_features, labels


class TripletTrafficDataset(Dataset):
    """
    Dataset PyTorch que retorna a tupla (Anchor, Positive, Negative)
    """
    def __init__(self, features, labels):
        # Normalização Z-score para estabilizar o aprendizado da rede neural
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) + 1e-8
        self.features = torch.tensor((features - mean) / std, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
        self.labels_set = set(labels)
        # Mapeia quais índices pertencem a qual classe para acelerar o sorteio
        self.label_to_indices = {label: np.where(labels == label)[0] for label in self.labels_set}

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        anchor_data = self.features[index]
        anchor_label = self.labels[index].item()

        # Seleciona um POSITIVO (mesma classe, amostra diferente)
        positive_index = index
        while positive_index == index:
            positive_index = random.choice(self.label_to_indices[anchor_label])
        positive_data = self.features[positive_index]

        # Seleciona um NEGATIVO (classe oposta)
        negative_label = random.choice(list(self.labels_set - set([anchor_label])))
        negative_index = random.choice(self.label_to_indices[negative_label])
        negative_data = self.features[negative_index]

        return anchor_data, positive_data, negative_data