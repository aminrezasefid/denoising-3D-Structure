import torch
from torch_geometric.transforms import Compose
from torch_geometric.datasets import QM9 as QM9_geometric
from torch_geometric.nn.models.schnet import qm9_target_dict
from tqdm import tqdm
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip,
                                  Data)
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem import AllChem                  

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

atomrefs = {
    6: [0., 0., 0., 0., 0.],
    7: [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    8: [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    9: [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702
    ],
    10: [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
    11: [0., 0., 0., 0., 0.],
}

class QM9(QM9_geometric): # dataset downloding and others processing  
    def __init__(self, root, transform=None, dataset_arg=None):
        # assert dataset_arg is not None, (
        #     "Please pass the desired property to "
        #     'train on via "dataset_arg". Available '
        #     f'properties are {", ".join(qm9_target_dict.values())}.'
        # )
        if dataset_arg is not None:
            self.label = dataset_arg
            label2idx = dict(zip(qm9_target_dict.values(), qm9_target_dict.keys()))
            self.label_idx = label2idx[self.label]

            if transform is None:
                transform = self._filter_label
            else:
                transform = Compose([transform, self._filter_label])

        super(QM9, self).__init__(root, transform=transform)

    def get_atomref(self, max_z=100):
        atomref = self.atomref(self.label_idx)
        if atomref is None:
            return None
        if atomref.size(0) != max_z:
            tmp = torch.zeros(max_z).unsqueeze(1)
            idx = min(max_z, atomref.size(0))
            tmp[:idx] = atomref[:idx]
            return tmp
        return atomref

    def _filter_label(self, batch): #return batch with only targets
        batch.y = batch.y[:, self.label_idx].unsqueeze(1)
        return batch

    def download(self):
        super(QM9, self).download()
    # def process(self):
    #     super(QM9, self).process()
    #     suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
    #                                sanitize=False)
    #     data_list= torch.load(self.processed_paths[0])

    #     for i, mol in enumerate(tqdm(suppl)):
    #         name = mol.GetProp('_Name')
    #         if name not in data_list[0].name:
    #             continue
            
            

    def process(self):
        try:
            import rdkit
            from rdkit import Chem
            from rdkit.Chem.rdchem import HybridizationType
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            rdkit = None

        if rdkit is None:
            print(("Using a pre-processed version of the dataset. Please "
                   "install 'rdkit' to alternatively process the raw data."),
                  file=sys.stderr)

            data_list = torch.load(self.raw_paths[0])
            data_list = [Data(**data_dict) for data_dict in data_list]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            torch.save(self.collate(data_list), self.processed_paths[0])
            return

        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        with open(self.raw_paths[1], 'r') as f:
            target = f.read().split('\n')[1:-1]
            target = [[float(x) for x in line.split(',')[1:20]]
                      for line in target]
            target = torch.tensor(target, dtype=torch.float)
            target = torch.cat([target[:, 3:], target[:, :3]], dim=-1)
            target = target * conversion.view(1, -1)

        with open(self.raw_paths[2], 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=True)

        data_list = []
        params = Chem.SmilesParserParams()
        params.removeHs = False
        counter=0
        ps = rdDistGeom.ETKDGv3()
        ps.maxIters=100
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip or mol==None:
                continue
            
            N = mol.GetNumAtoms()

            pos = suppl.GetItemText(i).split('\n')[4:4 + N]
            pos = [[float(x) for x in line.split()[:3]] for line in pos]
            pos = torch.tensor(pos, dtype=torch.float)

            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

            z = torch.tensor(atomic_number, dtype=torch.long)

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type,
                                  num_classes=len(bonds)).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N).tolist()

            x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                              dtype=torch.float).t().contiguous()
            x = torch.cat([x1.to(torch.float), x2], dim=-1)

            y = target[i].unsqueeze(0)
            name = mol.GetProp('_Name')
            smi=Chem.MolToSmiles(mol)
            noisy_mol=Chem.MolFromSmiles(smi,params)
            smi2=Chem.MolToSmiles(noisy_mol)
            generated=-1
            generated=rdDistGeom.EmbedMolecule(noisy_mol,ps)
            if generated==-1:
                continue
            try:
                mapping = noisy_mol.GetSubstructMatch(mol)
                noisy_mol = Chem.RenumberAtoms(noisy_mol, mapping)
                Chem.rdMolAlign.AlignMol(noisy_mol,mol)
            except:
                continue
            # data = Data(x=x, z=z, pos=pos, edge_index=edge_index,
            #             edge_attr=edge_attr, y=y, name=name, idx=i,mol=mol,noisy_mol=noisy_mol)
            
            data = Data(x=x, z=z, pos=pos, edge_index=edge_index,
                        edge_attr=edge_attr, name=name, idx=i,mol=mol,noisy_mol=noisy_mol)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        print(counter)
        torch.save(self.collate(data_list), self.processed_paths[0])

