import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.smiles_parser import smiles_to_graph
from src.feature_engineering import extract_features


class MolecularSolubilityDataset(Dataset):
    """
    Dataset for molecular solubility prediction from SMILES.
    """
    def __init__(self, smiles_list, solubility_values, feature_method='basic'):
        self.smiles = smiles_list
        self.solubility = solubility_values
        self.feature_method = feature_method
        self.features = []
        
        # Precompute features
        for smiles in self.smiles:
            mol_graph = smiles_to_graph(smiles)
            features = extract_features(mol_graph, method=feature_method)
            self.features.append(features)
        
        self.features = torch.tensor(np.array(self.features), dtype=torch.float32) # mb half precision is enough?
        self.solubility = torch.tensor(self.solubility, dtype=torch.float32).view(-1, 1)
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        return self.features[idx], self.solubility[idx]


def load_esol_dataset(file_path, test_size=0.2, val_size=0.2, feature_method='basic', random_state=42):
    """
    Load the ESOL dataset and split it into train, validation, and test sets.
    
    Args:
        file_path: Path to the ESOL dataset CSV file
        test_size: Fraction of data to use for testing
        val_size: Fraction of training data to use for validation
        feature_method: Method for feature extraction
        random_state: Random seed for reproducibility
        
    Returns:
        train_loader, val_loader, test_loader, feature_dim, scaler
    """

    df = pd.read_csv(file_path)
    
    smiles_col = "smiles"  
    solubility_col = "measured log solubility in mols per litre"  
    
    X = df[smiles_col].values
    y = df[solubility_col].values
    
    # Split into train+val and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state
    )
    
    # Normalize the values. Optional.
    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).ravel()
    
    train_dataset = MolecularSolubilityDataset(X_train, y_train_scaled, feature_method)
    val_dataset = MolecularSolubilityDataset(X_val, y_val_scaled, feature_method)
    test_dataset = MolecularSolubilityDataset(X_test, y_test_scaled, feature_method)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    feature_dim = train_dataset.features.shape[1]
    
    return train_loader, val_loader, test_loader, feature_dim, scaler
