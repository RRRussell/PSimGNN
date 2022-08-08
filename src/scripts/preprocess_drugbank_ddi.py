from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import networkx as nx
import re
import os
import utils
from collections import defaultdict

hybridization_to_int = {
    Chem.rdchem.HybridizationType.S: 0,
    Chem.rdchem.HybridizationType.SP: 1,
    Chem.rdchem.HybridizationType.SP2: 2,
    Chem.rdchem.HybridizationType.SP3: 3,
    Chem.rdchem.HybridizationType.SP3D: 4,
    Chem.rdchem.HybridizationType.SP3D2: 5,
    Chem.rdchem.HybridizationType.UNSPECIFIED:6
}


def mol_to_nxgraph(drug_mol, edge_atts=False):
    fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    feats = chem_feature_factory.GetFeaturesForMol(drug_mol)
    # assert len(drug_mol.GetConformers()) == 1
    # geom = drug_mol.GetConformers()[0].GetPositions()

    nx_graph = nx.Graph()
    for i in range(drug_mol.GetNumAtoms()):
        atom = drug_mol.GetAtomWithIdx(i)

        nx_graph.add_node(i, atom_type=atom.GetSymbol(), aromatic=atom.GetIsAromatic(), acceptor=0,
                          donor=0, hybridization=hybridization_to_int[atom.GetHybridization()],
                          num_h=atom.GetTotalNumHs())
    for i in range(len(feats)):
        if feats[i].GetFamily() == 'Donor':
            node_list = feats[i].GetAtomIds()
            for i in node_list:
                nx_graph.node[i]['donor'] = 1
        elif feats[i].GetFamily() == 'Acceptor':
            node_list = feats[i].GetAtomIds()
            for i in node_list:
                nx_graph.node[i]['acceptor'] = 1
    # Read Edges
    for i in range(drug_mol.GetNumAtoms()):
        for j in range(drug_mol.GetNumAtoms()):
            e_ij = drug_mol.GetBondBetweenAtoms(i, j)
            if e_ij is not None:
                if not edge_atts:
                    nx_graph.add_edge(i, j)
                else:
                    nx_graph.add_edge(i, j, b_type=e_ij.GetBondType())
    return nx_graph

def generate_gexf_from_sdf(sdf_file, dataset_name=None, nodes=None, edge_atts=False):
    if not dataset_name:
        dataset_path =  os.path.join(utils.get_data_path(), "DrugBank")
    elif dataset_name == "ddi_snap":
        dataset_path = os.path.join(utils.get_data_path(), "DrugBank", "DDI_datasets", "drugs_snap")
    elif dataset_name == "ddi_deepddi":
        dataset_path = os.path.join(utils.get_data_path(), "DrugBank", "DDI_datasets", "drugs_deepddi")
    else:
        raise ValueError("{} not supported".format(dataset_name))
    klepto_path = os.path.join(dataset_path, "klepto")
    utils.create_dir_if_not_exists(dataset_path)
    utils.create_dir_if_not_exists(klepto_path)
    drug_mols = Chem.SDMolSupplier(sdf_file)
    skipped = []
    drug_grp_count = defaultdict(int)
    drug_to_grp_dict = {}
    for i, drug_mol in enumerate(drug_mols):
        if i % 25 == 0:
            print("processed {} graphs".format(i))
        if drug_mol is None:
            print("rdkit did not parse drug ", i)
            skipped.append(i)
            continue
        if not drug_mol.HasProp("DRUGBANK_ID"):
            raise AssertionError("DRUGBANK_ID attribute not found")
        db_id = drug_mol.GetProp("DRUGBANK_ID")

        if nodes and db_id not in nodes:
            continue
        nx_graph = mol_to_nxgraph(drug_mol, edge_atts)


        nx_graph.graph["db_id"] = db_id

        if not drug_mol.HasProp("DRUG_GROUPS"):
            raise AssertionError("DRUG_GROUPS attribute not found")
        db_grp = drug_mol.GetProp("DRUG_GROUPS")
        drug_grp_count[db_grp] += 1
        nx_graph.graph["db_grp"] = db_grp.split(';')
        drug_to_grp_dict[db_id] = {"db_grp": db_grp.split(';'), "db_id": db_id}
        nx.readwrite.write_gexf(nx_graph, os.path.join(dataset_path, db_id + ".gexf"))
    utils.save(dict(drug_grp_count), os.path.join(klepto_path, "drug_group_count.klepto"))
    utils.save(drug_to_grp_dict, os.path.join(klepto_path, "graph_data.klepto"))
    return skipped, drug_grp_count, drug_to_grp_dict



def check_unqiue_nodes(file_path, parse_edges_func):
    drug_ids = set()
    edge_list = []
    edge_types = set()
    with open(file_path, 'r') as f:
        for i, line in enumerate(f.readlines()):

            edges, edge_type = parse_edges_func(line)
            if edges:
                drug_ids.add(edges[0])
                drug_ids.add(edges[1])
                edge_list.append((edges[0], edges[1]))
                edge_types.add(edge_type)

    return drug_ids, edge_types, edge_list

def parse_edges_biosnap(line):
    return line.rstrip('\n').split('\t'), None

def parse_edges_pnas(line):
    line = line.split(',')[0]
    pattern = 'DB[0-9]{5}'
    drugs = list(set(re.findall(pattern, line)))
    edge = None
    if drugs:
        if line.find(drugs[0]) < line.find(drugs[1]):
            drug1, drug2 = drugs[0], drugs[1]
        else:
            drug1, drug2 = drugs[1], drugs[0]
        temp = re.sub(drug1, 'D1', line)
        edge = re.sub(drug2, 'D2', temp)
    return drugs, edge

if __name__ == "__main__":
    from collections import Counter

    m = Chem.MolFromSmiles('[Ca++].CC([O-])=O.CC([O-])=O')
    nx_graph = mol_to_nxgraph(m)
    file_path = r"D:\Ken\University\Research Please\Datasets\DDI_Datasets\ChCh-Miner_durgbank-chem-chem.tsv"
    ddi_drugs_biosnap, edge_types, edge_list = check_unqiue_nodes(file_path, parse_edges_biosnap)
    file_path = r"D:\Ken\University\Research Please\Datasets\DDI_Datasets\pnas_ddi.csv"
    ddi_drugs_pnas, edge_types, edge_list = check_unqiue_nodes(file_path, parse_edges_pnas)

    file_path = r"D:\Ken\University\Research Please\Datasets\DDI_Datasets\drugbank_all_structures.sdf\structures.sdf"
    skipped, drug_grp_count, drug_to_grp_dict = generate_gexf_from_sdf(file_path, dataset_name="ddi_deepddi", nodes=ddi_drugs_pnas)

    print("skipped: ", skipped)
    print("drug_grp_count: ", drug_grp_count)




    # count = 0
    # for drug in ddi_drugs_pnas:
    #     if drug in ddi_drugs_biosnap:
    #         count += 1
    # print(count)

    #utils.save(ddi_drugs_pnas, os.path.join(klepto_path, "pnas_ddi_drug_set.klepto"))
    print("Done")
