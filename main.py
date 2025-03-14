import argparse
import torch
import numpy as np

from src.datasets import load_esol_dataset
from src.models import get_model
from src.training import train_regression_model, evaluate_model
from src.utils import plot_training_history, plot_predictions, compare_feature_methods


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    feature_methods = args.feature_methods.split(',')
    print(f"Featurizing methods to use: {feature_methods}")
    
    results = {}
    
    for method in feature_methods:
        print(f"\n\n{'=' * 50}")
        print(f"Training with feature method: {method}")
        print(f"{'=' * 50}\n")
        
        train_loader, val_loader, test_loader, feature_dim, scaler = load_esol_dataset(
            args.data_path, 
            test_size=args.test_size,
            val_size=args.val_size,
            feature_method=method,
            random_state=args.seed
        )
        
        print(f"Feature dimension: {feature_dim}")
        
        model = get_model(feature_dim, model_type=args.model_type)
        
        model, history = train_regression_model(
            model,
            train_loader,
            val_loader,
            epochs=args.epochs,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        if args.plot:
            plot_training_history(history)
        
        metrics = evaluate_model(model, test_loader, scaler)
        results[method] = metrics
        
        print("\nTest Results:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        if args.plot:
            model.eval()
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for features, targets in test_loader:
                    outputs = model(features)
                    all_preds.extend(outputs.numpy().flatten())
                    all_targets.extend(targets.numpy().flatten())

            # print(all_targets)
            # print(all_preds)
            
            plot_predictions(
                np.array(all_targets),
                np.array(all_preds),
                title=f"Predictions using {method} features"
            )
    
    if len(feature_methods) > 1:
        comparison = compare_feature_methods(results)
        print("\nFeature Method Comparison:")
        print(comparison)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Molecular Solubility Prediction")
    
    parser.add_argument("--data_path", type=str, default="data/delaney-processed.csv",
                        help="Path to the ESOL dataset")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data to use for testing")
    parser.add_argument("--val_size", type=float, default=0.2,
                        help="Fraction of training data to use for validation")
    
    parser.add_argument("--feature_methods", type=str, default="basic,morgan,global",
                        help="Comma-separated list of feature methods to try")
    
    parser.add_argument("--model_type", type=str, default="simple_regression",
                        help="Type of model to use")
    
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="L2 regularization parameter")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--plot", action="store_true",
                        help="Whether to plot results")
    
    args = parser.parse_args()
    main(args)
