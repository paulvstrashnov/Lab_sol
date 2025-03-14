import numpy as np
from src.smiles_parser import SMILESParser

class FeatureExtractor:
    """
    Extracts features from a molecular graph for machine learning.
    """
    # Simple periodic table properties
    ATOMIC_MASSES = {
        'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998,
        'P': 30.974, 'S': 32.065, 'Cl': 35.453, 'Br': 79.904, 'I': 126.904
    }
    
    ELECTRONEGATIVITY = {
        'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
        'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66
    }
    
    ATOMIC_RADII = {  # in picometers
        'H': 25, 'C': 70, 'N': 65, 'O': 60, 'F': 50,
        'P': 100, 'S': 100, 'Cl': 100, 'Br': 115, 'I': 140
    }
    
    def __init__(self, mol_graph):
        self.graph = mol_graph
        
    def get_atom_features(self):
        """Extract basic atom-level features."""
        n_atoms = len(self.graph.atoms)
        features = np.zeros((n_atoms, 4))  # 4 features per atom
        
        for i, atom_type in enumerate(self.graph.atoms):
            atomic_number = SMILESParser.ATOM_SYMBOLS.get(atom_type, 0)
            features[i, 0] = atomic_number / 100  # Normalize
            features[i, 1] = self.graph.get_degree(i) / 6  # Normalize
            features[i, 3] = self.ELECTRONEGATIVITY.get(atom_type, 0) / 4  # Normalize
            
        return features
    
    def aggregate_features(self, atom_features, method='sum'):
        """
        Aggregate atom features to create a single feature vector for the molecule.
        
        Args:
            atom_features: Numpy array of atom features
            method: Aggregation method ('sum', 'mean', 'max')
            
        Returns:
            A feature vector for the entire molecule
        """
        if method == 'sum':
            return np.sum(atom_features, axis=0)
        elif method == 'mean':
            return np.mean(atom_features, axis=0)
        elif method == 'max':
            return np.max(atom_features, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def morgan_like_update(self, atom_features, iterations=2):
        """
        Update atom features based on their neighbors, similar to Morgan algorithm.
        
        Args:
            atom_features: Initial atom features
            iterations: Number of iterations to run
            
        Returns:
            Updated atom features
        """
        pass
    
    def extract_global_features(self):
        """
        Extract simple global molecular features as a baseline.
        
        Returns:
            A feature vector with global molecular properties
        """

        n_atoms = len(self.graph.atoms)
        n_bonds = sum(len(neighbors) for neighbors in self.graph.bonds.values()) // 2
        avg_degree = n_bonds * 2 / max(1, n_atoms)
        mol_weight = sum(self.ATOMIC_MASSES.get(atom, 0) for atom in self.graph.atoms)
        
        return np.array([n_atoms, n_bonds, avg_degree, mol_weight])


def extract_features(mol_graph, method='basic'):
    """
    Extract features from a molecular graph based on the specified method.
    
    Args:
        smiles: Original SMILES string (unused in basic implementation)
        mol_graph: Molecular graph object
        method: Feature extraction method
        
    Returns:
        A feature vector for the molecule
    """
    extractor = FeatureExtractor(mol_graph)
    
    if method == 'basic':
        atom_features = extractor.get_atom_features()
        return extractor.aggregate_features(atom_features, 'sum')
    
    elif method == 'morgan':
        atom_features = extractor.get_atom_features()
        updated_features = extractor.morgan_like_update(atom_features, iterations=2)
        return extractor.aggregate_features(updated_features, 'sum')
    
    elif method == 'global':
        return extractor.extract_global_features()
    
    else:
        raise ValueError(f"Unknown feature extraction method: {method}")


# if __name__ == "__main__":

#     smiles = "CCO"
#     mol_graph = SMILESParser.parse(smiles)
    
#     basic_features = extract_features(mol_graph, method='basic')
#     morgan_features = extract_features(mol_graph, method='morgan')
#     global_features = extract_features(mol_graph, method='global')
    
#     print("Basic Features:", basic_features)
#     print("Morgan Features:", morgan_features)
#     print("Global Features:", global_features)