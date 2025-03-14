import networkx as nx

class MolecularGraph:
    """
    Wrapper for NetworkX graph to represent a molecular structure.
    """
    def __init__(self):
        self.graph = nx.Graph()
    
    def add_atom(self, atom_type):
        """Add an atom to the graph and return its index."""
        atom_idx = len(self.graph.nodes)
        self.graph.add_node(atom_idx, atom_type=atom_type)
        return atom_idx
    
    def add_bond(self, atom1_idx, atom2_idx, bond_type=1):
        """Add a bond between two atoms with specified bond type."""
        self.graph.add_edge(atom1_idx, atom2_idx, bond_type=bond_type)
    
    @property
    def atoms(self):
        """Get a list of all atom types in the graph."""
        return [data['atom_type'] for _, data in self.graph.nodes(data=True)]
    
    def get_neighbors(self, atom_idx):
        """Get all neighbors of an atom."""
        return list(self.graph.neighbors(atom_idx))
    
    def get_bond_type(self, atom1_idx, atom2_idx):
        """Get the bond type between two atoms."""
        if self.graph.has_edge(atom1_idx, atom2_idx):
            return self.graph[atom1_idx][atom2_idx]['bond_type']
        return 0
    
    def get_degree(self, atom_idx):
        """Get the degree (number of bonds) of an atom."""
        return self.graph.degree(atom_idx)
    
    def __str__(self):
        return f"MolecularGraph with {len(self.atoms)} atoms and {self.graph.number_of_edges()} bonds"


class SMILESParser:
    """
    Simplified SMILES parser that only handles linear molecules with single bonds.
    """

    ATOM_SYMBOLS = {
        'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 
        'Cl': 17, 'Br': 35, 'I': 53
    }
    
    def __init__(self):
        pass
    
    def parse(self, smiles):
        """
        Parses a SMILES string and return a MolecularGraph.
        Doesn't work with aromatics, two-letter elements like Cl, Br, and non-linear molecules
        """
        graph = MolecularGraph()
        
        previous_atom_idx = None

        atom_symbols = [char for char in smiles if char in self.ATOM_SYMBOLS]
        
        for atom_symbol in atom_symbols:
            
            atom_idx = graph.add_atom(atom_symbol)
            
            if previous_atom_idx is not None:
                graph.add_bond(previous_atom_idx, atom_idx)
            
            previous_atom_idx = atom_idx
        
        return graph


def smiles_to_graph(smiles):
    """Utility function to convert SMILES to a molecular graph."""
    parser = SMILESParser()
    return parser.parse(smiles)


if __name__ == "__main__":
    graph = smiles_to_graph("CCO")
    print(graph)
    print(graph.atoms)
    print(graph.get_neighbors(0))
    print(graph.get_bond_type(0, 1))
    print(graph.get_degree(1))