import torch.nn as nn
import torch.nn.functional as F


class SimpleRegressionModel(nn.Module):
    """
    A simple one-layer neural network for solubility prediction as a regression task.
    """
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class SimpleClassificationModel(nn.Module):
    """
    A simple one-layer neural network for solubility prediction as a classification task.
    """
    pass


def get_model(feature_dim, model_type='simple_regression'):
    """
    Factory function to create the appropriate model.
    
    Args:
        feature_dim: Dimension of the input features
        model_type: Type of model to create
        
    Returns:
        A PyTorch model
    """
    if model_type == 'simple_regression':
        return SimpleRegressionModel(feature_dim)
    elif model_type == 'simple_classification':
        return SimpleClassificationModel(feature_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
