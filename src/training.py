import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from sklearn.metrics import mean_squared_error, r2_score


def train_regression_model(model, train_loader, val_loader, epochs=100, lr=0.001, weight_decay=1e-4):
    """
    Train a model for solubility prediction.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization parameter
        
    Returns:
        Trained model and training history
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_r2': [],
        'val_r2': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        start_time = time.time()
        
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for features, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
            train_preds.extend(outputs.detach().numpy())
            train_targets.extend(targets.numpy())
        
        train_loss /= len(train_loader.dataset)
        train_r2 = r2_score(train_targets, train_preds)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for features, targets in val_loader:
                outputs = model(features)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * features.size(0)
                val_preds.extend(outputs.numpy())
                val_targets.extend(targets.numpy())
        
        val_loss /= len(val_loader.dataset)
        val_r2 = r2_score(val_targets, val_preds)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - "
              f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - "
              f"Train R²: {train_r2:.4f} - Val R²: {val_r2:.4f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


def evaluate_model(model, test_loader, scaler=None):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        scaler: StandardScaler used to normalize targets (for inverse transform)
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            outputs = model(features)
            test_preds.extend(outputs.numpy())
            test_targets.extend(targets.numpy())
    
    mse = mean_squared_error(test_targets, test_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_targets, test_preds)
    
    if scaler is not None:
        test_preds_orig = scaler.inverse_transform(np.array(test_preds).reshape(-1, 1)).ravel()
        test_targets_orig = scaler.inverse_transform(np.array(test_targets).reshape(-1, 1)).ravel()
        mse_orig = mean_squared_error(test_targets_orig, test_preds_orig)
        rmse_orig = np.sqrt(mse_orig)
        r2_orig = r2_score(test_targets_orig, test_preds_orig)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mse_orig': mse_orig,
            'rmse_orig': rmse_orig,
            'r2_orig': r2_orig
        }
    else:
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
    
    return metrics
