import torch
import torch.nn.functional as F
from typing import List

class ZeroPrintFingerprintHelper:
    def __init__(self, config, accelerator=None):
        self.config = config
        self.accelerator = accelerator
        self.similarity_metric = config.get('similarity_metric', 'cosine')
    
    def compute_similarity(self, fingerprint1: torch.Tensor, fingerprint2: torch.Tensor) -> float:
        """
        Compute similarity between two fingerprints using the configured metric.
        
        Args:
            fingerprint1: First fingerprint tensor
            fingerprint2: Second fingerprint tensor
            
        Returns:
            float: Similarity score in [0,1] range
        """
        # Ensure both fingerprints are tensors on the correct device
        device = self.accelerator.device if self.accelerator is not None else 'cpu'
        
        if not isinstance(fingerprint1, torch.Tensor):
            fingerprint1 = torch.tensor(fingerprint1, dtype=torch.float32, device=device)
        else:
            fingerprint1 = fingerprint1.to(device)
            
        if not isinstance(fingerprint2, torch.Tensor):
            fingerprint2 = torch.tensor(fingerprint2, dtype=torch.float32, device=device)
        else:
            fingerprint2 = fingerprint2.to(device)
        
        # Flatten the tensors if they are multi-dimensional
        fingerprint1 = fingerprint1.flatten()
        fingerprint2 = fingerprint2.flatten()
        
        if self.similarity_metric == 'cosine':
            # Cosine similarity: [-1,1] -> [0,1]
            cosine_sim = F.cosine_similarity(fingerprint1.unsqueeze(0), fingerprint2.unsqueeze(0))
            return ((cosine_sim + 1) / 2).item()
            
        elif self.similarity_metric == 'correlation':
            # Pearson correlation coefficient: [-1,1] -> [0,1]
            mean1, mean2 = fingerprint1.mean(), fingerprint2.mean()
            centered1 = fingerprint1 - mean1
            centered2 = fingerprint2 - mean2
            
            numerator = (centered1 * centered2).sum()
            denominator = (centered1.pow(2).sum() * centered2.pow(2).sum()).sqrt()
            
            if denominator == 0:
                correlation = torch.tensor(0.0, device=device)
            else:
                correlation = numerator / denominator
                
            return ((correlation + 1) / 2).item()
            
        elif self.similarity_metric == 'euclidean':
            # Euclidean distance -> similarity: [0,inf] -> [0,1]
            distance = torch.norm(fingerprint1 - fingerprint2, p=2)
            max_distance = torch.norm(fingerprint1) + torch.norm(fingerprint2)
            if max_distance == 0:
                return 1.0
            similarity = 1 - (distance / max_distance)
            return max(0.0, similarity.item())
            
        elif self.similarity_metric == 'manhattan':
            # Manhattan distance -> similarity: [0,inf] -> [0,1]
            distance = torch.norm(fingerprint1 - fingerprint2, p=1)
            max_distance = torch.norm(fingerprint1, p=1) + torch.norm(fingerprint2, p=1)
            if max_distance == 0:
                return 1.0
            similarity = 1 - (distance / max_distance)
            return max(0.0, similarity.item())
            
        elif self.similarity_metric == 'dot_product':
            # Normalized dot product: [0,1]
            norm1 = torch.norm(fingerprint1)
            norm2 = torch.norm(fingerprint2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            normalized_dot = torch.dot(fingerprint1, fingerprint2) / (norm1 * norm2)
            return max(0.0, normalized_dot.item())
            
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
    
    def compute_cosine_similarity(self, fingerprint1: torch.Tensor, fingerprint2: torch.Tensor) -> float:
        """
        Compute cosine similarity between two fingerprints and scale to [0,1] range.
        
        Args:
            fingerprint1: First fingerprint tensor
            fingerprint2: Second fingerprint tensor
            
        Returns:
            float: Cosine similarity score in [0,1] range
        """
        # Ensure both fingerprints are tensors
        if not isinstance(fingerprint1, torch.Tensor):
            fingerprint1 = torch.tensor(fingerprint1)
        if not isinstance(fingerprint2, torch.Tensor):
            fingerprint2 = torch.tensor(fingerprint2)
        
        # Flatten the tensors if they are multi-dimensional
        fingerprint1 = fingerprint1.flatten()
        fingerprint2 = fingerprint2.flatten()
        
        # Compute cosine similarity and scale from [-1,1] to [0,1]
        cosine_sim = F.cosine_similarity(fingerprint1.unsqueeze(0), fingerprint2.unsqueeze(0))
        similarity_score = (cosine_sim + 1) / 2
        
        return similarity_score.item()
