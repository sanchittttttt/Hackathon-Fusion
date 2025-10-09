import torch
import torch.nn as nn


class BindingMLP(nn.Module):
    def __init__(self, protein_dim=480, molecule_dim=2048):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(protein_dim + molecule_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        print(f"✅ MLP ready ({sum(p.numel() for p in self.parameters()):,} params)")
    
    def forward(self, protein, molecule):
        x = torch.cat([protein, molecule], dim=1)
        return self.network(x)
    
    def mc_dropout_predict(self, protein, molecule, n_samples=30):
        self.train()  # Keep dropout ON during inference
        
        preds = []
        for _ in range(n_samples):
            preds.append(self.forward(protein, molecule))
        preds = torch.stack(preds, dim=0)  # Shape: (n_samples, batch_size, 1)
        
        # mean prediction and std deviation (used for confidence-aware ranking)
        mean_pred = preds.mean(dim=0).squeeze(-1)  # Shape: (batch_size,)
        std_pred = preds.std(dim=0).squeeze(-1)  # Shape: (batch_size,)
        
        return mean_pred, std_pred  # rename in caller for confidence-aware ranking
