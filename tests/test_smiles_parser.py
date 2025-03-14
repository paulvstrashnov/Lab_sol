import unittest
from src.smiles_parser import smiles_to_graph, SMILESParser, MolecularGraph


class TestMolecularGraph(unittest.TestCase):
    """Tests for the MolecularGraph class functionality"""
    
    def test_graph_initialization(self):
        graph = MolecularGraph()
        self.assertEqual(len(graph.atoms), 0)
        self.assertEqual(graph.graph.number_of_edges(), 0)
        
    def test_add_atom(self):
        graph = MolecularGraph()
        idx = graph.add_atom("C")
        self.assertEqual(idx, 0)
        self.assertEqual(graph.atoms, ["C"])
        
        idx = graph.add_atom("O")
        self.assertEqual(idx, 1)
        self.assertEqual(graph.atoms, ["C", "O"])
        
    def test_add_bond(self):
        graph = MolecularGraph()
        graph.add_atom("C")
        graph.add_atom("O")
        graph.add_bond(0, 1, bond_type=1)
        
        self.assertTrue(graph.graph.has_edge(0, 1))
        self.assertEqual(graph.get_bond_type(0, 1), 1)
        
        graph.add_bond(0, 1, bond_type=2)  # Change bond type
        self.assertEqual(graph.get_bond_type(0, 1), 2)
        
    def test_get_neighbors(self):
        graph = MolecularGraph()
        graph.add_atom("C")  # 0
        graph.add_atom("O")  # 1
        graph.add_atom("N")  # 2
        graph.add_bond(0, 1)
        graph.add_bond(0, 2)
        
        neighbors = graph.get_neighbors(0)
        self.assertEqual(set(neighbors), {1, 2})
        self.assertEqual(graph.get_neighbors(1), [0])
        
    def test_get_degree(self):
        graph = MolecularGraph()
        graph.add_atom("C")  # 0
        graph.add_atom("O")  # 1
        graph.add_atom("N")  # 2
        graph.add_atom("F")  # 3
        
        graph.add_bond(0, 1)
        graph.add_bond(0, 2)
        graph.add_bond(0, 3)
        
        self.assertEqual(graph.get_degree(0), 3)
        self.assertEqual(graph.get_degree(1), 1)


class TestSMILESParser(unittest.TestCase):
    """Tests for the SMILES parser functionality"""
    
    def test_single_atoms(self):
        atoms = ["C", "O", "N", "F", "P", "S"]
        for atom in atoms:
            molecule = smiles_to_graph(atom)
            self.assertEqual(len(molecule.atoms), 1)
            self.assertEqual(molecule.atoms[0], atom)
            self.assertEqual(molecule.graph.number_of_edges(), 0)
    
    def test_linear_molecules(self):
        methanol = smiles_to_graph("CO")
        self.assertEqual(methanol.atoms, ["C", "O"])
        self.assertEqual(methanol.get_bond_type(0, 1), 1)
        
        propane = smiles_to_graph("CCC")
        self.assertEqual(len(propane.atoms), 3)
        self.assertEqual(propane.atoms, ["C", "C", "C"])
        self.assertEqual(propane.get_bond_type(0, 1), 1)
        self.assertEqual(propane.get_bond_type(1, 2), 1)
        self.assertFalse(propane.graph.has_edge(0, 2))
        
    def test_bond_types(self):
        co2 = smiles_to_graph("O=C=O")
        self.assertEqual(len(co2.atoms), 3)
        self.assertEqual(co2.atoms, ["O", "C", "O"])
        self.assertEqual(co2.get_bond_type(0, 1), 2)  # Double bond
        self.assertEqual(co2.get_bond_type(1, 2), 2)  # Double bond
        
        ethylene = smiles_to_graph("C=C")
        self.assertEqual(len(ethylene.atoms), 2)
        self.assertEqual(ethylene.get_bond_type(0, 1), 2)
        
        acetylene = smiles_to_graph("C#C")
        self.assertEqual(len(acetylene.atoms), 2)
        self.assertEqual(acetylene.get_bond_type(0, 1), 3)
        
    def test_branched_molecules(self):
        isobutane = smiles_to_graph("CC(C)C")
        self.assertEqual(len(isobutane.atoms), 4)
        self.assertEqual(isobutane.atoms, ["C", "C", "C", "C"])
        self.assertEqual(isobutane.get_bond_type(0, 1), 1)
        self.assertEqual(isobutane.get_bond_type(1, 2), 1)
        self.assertEqual(isobutane.get_bond_type(1, 3), 1)
        
        tert_butanol = smiles_to_graph("CC(C)(C)O")
        self.assertEqual(len(tert_butanol.atoms), 5)
        self.assertEqual(tert_butanol.atoms, ["C", "C", "C", "C", "O"])
        self.assertTrue(tert_butanol.graph.has_edge(1, 4))  # Bond to oxygen
    
    def test_ring_structures(self):
        cyclohexane = smiles_to_graph("C1CCCCC1")
        self.assertEqual(len(cyclohexane.atoms), 6)
        self.assertTrue(cyclohexane.graph.has_edge(0, 5))  # Ring closure
        
        benzene = smiles_to_graph("c1ccccc1")
        self.assertEqual(len(benzene.atoms), 6)
        self.assertTrue(benzene.graph.has_edge(0, 5))
        
    def test_complex_structures(self):
        glucose = smiles_to_graph("C1C(O)C(O)C(O)C(O)C1O")
        self.assertEqual(len(glucose.atoms), 12)
        
        aspirin = smiles_to_graph("CC(=O)OC1=CC=CC=C1C(=O)O")
        self.assertTrue(len(aspirin.atoms) > 10)


if __name__ == "__main__":
    unittest.main()