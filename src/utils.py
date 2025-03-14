import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def plot_training_history(history):
    """
    Plot the training and validation loss and R² scores.
    
    Args:
        history: Dictionary with training history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot R²
    ax2.plot(history['train_r2'], label='Train')
    ax2.plot(history['val_r2'], label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('R²')
    ax2.set_title('Training and Validation R²')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def plot_predictions(y_true, y_pred, title='Predicted vs. Actual Values'):
    """
    Plot predicted vs. actual values with a diagonal reference line.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        title: Plot title
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.7)
    
    # diagonal line
    lims = [
        min(min(y_true), min(y_pred)),
        max(max(y_true), max(y_pred))
    ]
    plt.plot(lims, lims, 'k--', alpha=0.5)
    
    # regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_true, p(y_true), "r--", alpha=0.5)
    
    # metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    plt.text(lims[0] + 0.1*(lims[1]-lims[0]), lims[1] - 0.1*(lims[1]-lims[0]), 
             f'RMSE = {rmse:.3f}\nR² = {r2:.3f}')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def compare_feature_methods(results_dict):
    """
    Compare performance of different feature engineering methods.
    
    Args:
        results_dict: Dictionary mapping feature method names to their metrics
    """
    methods = list(results_dict.keys())
    rmse_values = [results_dict[method]['rmse'] for method in methods]
    r2_values = [results_dict[method]['r2'] for method in methods]
    
    comparison_df = pd.DataFrame({
        'Method': methods,
        'RMSE': rmse_values,
        'R²': r2_values
    })
    
    comparison_df = comparison_df.sort_values('RMSE')
    
    return comparison_df