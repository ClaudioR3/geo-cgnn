import csv
import functools
import json
import os
import random
import warnings
import pickle
import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
import geo.utils as Utils


def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False, **kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    total_size = len(dataset)
    if train_ratio is None:
        assert val_ratio + test_ratio < 1
        train_ratio = 1 - val_ratio - test_ratio
        print('[Warning] train_ratio is None, using all training data.')
    else:
        assert train_ratio + val_ratio + test_ratio <= 1
    indices = list(range(total_size))
    if kwargs['train_size']:
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)
    if kwargs['test_size']:
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
    if kwargs['val_size']:
        valid_size = kwargs['val_size']
    else:
        valid_size = int(val_ratio * total_size)
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(
        indices[-(valid_size + test_size):-test_size])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(dataset, 
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, 
                              pin_memory=pin_memory,
                              drop_last=True)
    val_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn,
                            pin_memory=pin_memory,
                            drop_last=True)
    if return_test:
        test_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, 
                                 pin_memory=pin_memory,
                                 drop_last=True)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def collate_pool(batch):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    nodes = []
    edge_distance=[]
    edge_targets=[]
    edge_sources = []
    graph_indices = []
    node_counts = []
    targets = []
    combine_sets =[]
    plane_wave = []
    total_count = 0
    batch_cif_ids = []

    for i, (graph, target, cif_id) in enumerate(batch):

        # Numbering for each batch
        nodes.append(graph.nodes) 
        edge_distance.append(graph.distance) 
        edge_sources.append(graph.edge_sources + total_count) # source number of each edge
        edge_targets.append(graph.edge_targets + total_count) # target number of each edge
        combine_sets.append(graph.combine_sets)
        plane_wave.append(graph.plane_wave)
        node_counts.append(len(graph))
        targets.append(target)
        graph_indices += [i] * len(graph)
        total_count += len(graph)
        batch_cif_ids.append(cif_id)

    combine_sets=np.concatenate(combine_sets,axis=0)
    plane_wave=np.concatenate(plane_wave,axis=0)
    nodes = np.concatenate(nodes,axis=0)
    edge_distance = np.concatenate(edge_distance,axis=0)
    edge_sources = np.concatenate(edge_sources,axis=0)
    edge_targets = np.concatenate(edge_targets,axis=0)
    ginput = geo_CGNN_Input(nodes,edge_distance, edge_sources, edge_targets, graph_indices, node_counts,combine_sets,plane_wave)
    targets = torch.Tensor(targets)
    return ginput, targets, batch_cif_ids

#-------------------------------------------------
#--------------------ATOMGRAPH--------------------
#-------------------------------------------------
class AtomGraph(object):
    def __init__(self, graph,cutoff,N_shbf,N_srbf,n_grid_K,n_Gaussian):
        lattice, self.nodes, neighbors,volume = graph
        nei=neighbors[0] 
        distance=neighbors[1] 
        vector=neighbors[2] 
        n_nodes = len(self.nodes) 
        self.nodes = np.array(self.nodes, dtype=np.float32)
        self.edge_sources = np.concatenate([[i] * len(nei[i]) for i in range(n_nodes)])
        self.edge_targets=np.concatenate(nei)
        edge_vector = np.array(vector, dtype=np.float32)
        self.edge_index = np.concatenate([range(len(nei[i])) for i in range(n_nodes)])
        self.vectorij= edge_vector[self.edge_sources,self.edge_index]
        edge_distance = np.array(distance, dtype=np.float32)
        self.distance= edge_distance[self.edge_sources,self.edge_index]
        combine_sets=[]
        # gaussian radial
        N=n_Gaussian
        for n in range(1,N+1):
            phi=Phi(self.distance,cutoff)
            G=gaussian(self.distance,miuk(n,N,cutoff),betak(N,cutoff))
            combine_sets.append(phi*G)
        self.combine_sets=np.array(combine_sets, dtype=np.float32).transpose()

        # plane wave
        grid=n_grid_K
        kr=np.dot(self.vectorij,get_Kpoints_random(grid,lattice,volume).transpose()) 
        self.plane_wave=np.cos(kr)/np.sqrt(volume)

    def __len__(self):
        return len(self.nodes)

#-------------------------------------------------
#--------------------DATASET----------------------
#-------------------------------------------------
class CIFData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:

    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── id0.cif
    ├── id1.cif
    ├── ...

    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.

    atom_init.json: a JSON file that stores the initialization vector for each
    element.

    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.

    Parameters
    ----------

    root_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    disable_save_torch: bool
        Don't save torch files containing CIFData crystal graphs
    random_seed: int
        Random seed for shuffling the dataset

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """
    
    def __init__(self, root_dir, max_num_nbr=20, radius=16, 
                 cutoff=8, shbf=6, srbf=6, grid_K=4, n_gaussian=64,
                 disable_save_torch=False, random_seed=42, storage='file'):
        if storage=='file':
            self.storage = Utils.FileStorage(root_dir)
        elif storage=='ceph':
            self.storage = Utils.CephStorage(root_dir)
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.cutoff, self.shbf, self.srbf, self.grid_K, self.n_gaussian = cutoff, shbf, srbf, grid_K, n_gaussian
        self.disable_save_torch = disable_save_torch
        
        #-------- ID PROP CSV --------
        id_prop_file = str(os.path.join(self.root_dir, 'id_prop.csv'))
        try:
            reader = self.storage.read(id_prop_file)
        except Exception as e:
            warnings.warn('ERROR: id_prop.csv does not exist!')
            raise
        self.id_prop_data = [str(row).split(',') for row in reader.splitlines()]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        
        #-------- CONFIG JSON --------
        config_path = os.path.join(self.root_dir,'config_onehot.json')
        try:
            config = json.loads(self.storage.read(config_path))
        except Exception as e:
            print('WARNING: config_onehot.json does not exist!')
            config = self.build_config(config_path)
        self.config = config
        self.torch_data_path = os.path.join(self.root_dir, 'geo-cgnn')
        self.storage.mkdir(path=self.torch_data_path)
        

        
    def build_config(self, config_path):
        # 输入所有cubic.cif数据
        # 建立one-hot编码以及保存设置
        atoms=[]
        all_keys = [obj for obj in self.storage.objectslist(self.root_dir) # list_objects2S3(self.s3, self._bucket_name, self.root_dir)
                     if obj.endswith('.cif')]
        
        for key in tqdm(all_keys):
            crystal = Structure.from_str(
                self.storage.read(key),
                'cif'
            )
            atoms += list(crystal.atomic_numbers)
        unique_z = np.unique(atoms)
        num_z = len(unique_z)
        print('unique_z:', num_z)
        print('min z:', np.min(unique_z))
        print('max z:', np.max(unique_z))
        z_dict = {z:i for i, z in enumerate(unique_z)}
        # Configuration file
        config = dict()
        config["atomic_numbers"] = unique_z.tolist()
        config["node_vectors"] = np.eye(num_z,num_z).tolist() # One-hot encoding
        self.storage.mkdir(path=os.path.dirname(config_path))
        self.storage.write(file=config_path, body=json.dumps(config).encode('utf-8'))
        return config
    
    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        cif_id = str(int(cif_id.replace('ï»¿', '')))
        target = torch.Tensor([float(target)])
        try:
            reader = self.storage.read(
                os.path.join(self.torch_data_path, cif_id+'.pkl'), 
                decode = None)
            ag_data = pickle.loads(reader, encoding = "bytes")
        except ClientError as e:
            ag_data = self.process(cif_id)
        except FileNotFoundError as e:
            ag_data = self.process(cif_id)
        except pickle.UnpicklingError as ue:
            warnings.warn('{} - {}'.format(cif_id, str(ue)))
            ag_data = self.process(cif_id)
        #except pickle.KeyError as ke:
        #    warnings.warn('{} - {}'.format(cif_id, str(ke)))
        #    ag_data = self.process(cif_id)
            
        return ag_data, target, cif_id
    
    def process(self, cif_id: str):
        crystal = Structure.from_str(
            self.storage.read(
                os.path.join(self.root_dir, cif_id+'.cif')
            ),
            'cif')
        volume=crystal.lattice.volume
        coords=crystal.cart_coords
        lattice=crystal.lattice.matrix
        atoms=crystal.atomic_numbers
        atomnum=self.config['atomic_numbers']
        z_dict = {z:i for i, z in enumerate(atomnum)}
        one_hotvec=np.array(self.config["node_vectors"])
        atom_fea = np.vstack([one_hotvec[z_dict[atoms[i]]] for i in range(len(crystal))])

        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []

        for i,nbr in enumerate(all_nbrs):
            if len(nbr) < self.max_num_nbr:
                nbr_fea_idx.append(list(map(lambda x: x[2].tolist(), nbr)) +
                                    [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[0].coords.tolist(), nbr)) +
                       [[coords[i][0]+self.radius,coords[i][1],coords[i][2]]] * (self.max_num_nbr -len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2].tolist(),
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[0].coords.tolist(),
                                        nbr[:self.max_num_nbr])))
        atom_fea=atom_fea.tolist()

        nbr_subtract=[]
        nbr_distance=[]

        for i in range(len(nbr_fea)):
            if nbr_fea[i] != []:
                x=nbr_fea[i]-coords[:,np.newaxis,:][i]
                nbr_subtract.append(x)
                nbr_distance.append(np.linalg.norm(x, axis=1).tolist())
            else:
                nbr_subtract.append(np.array([]))
                nbr_distance.append(np.array([]))

        nbr_fea_idx = np.array(nbr_fea_idx) 
        
        graph = lattice,atom_fea,(nbr_fea_idx,nbr_distance,nbr_subtract),volume
        
        ag = AtomGraph(graph ,self.cutoff,self.shbf,self.srbf,self.grid_K,self.n_gaussian)
        
        if not self.disable_save_torch:
            self.storage.write( 
                    os.path.join(self.torch_data_path, cif_id+'.pkl'),
                    pickle.dumps(
                        ag, 
                        protocol=pickle.DEFAULT_PROTOCOL
                    )
            )
        return ag
        
    def clean_torch(self):
        self.storage.clean(self.torch_data_path)
        
# 构建torch的输入张量
class geo_CGNN_Input(object):
    def __init__(self,nodes,edge_distance,edge_sources, edge_targets, graph_indices, node_counts,combine_sets,plane_wave):
        self.nodes = torch.Tensor(nodes)
        self.edge_distance = torch.Tensor(edge_distance)
        self.edge_sources = torch.LongTensor(edge_sources)
        self.edge_targets = torch.LongTensor(edge_targets)
        self.graph_indices = torch.LongTensor(graph_indices)
        self.node_counts = torch.Tensor(node_counts)
        self.combine_sets=torch.Tensor(combine_sets)
        self.plane_wave=torch.Tensor(plane_wave)

    def __len__(self):
        return self.nodes.size(0)
                        
def a_SBF(alpha,l,n,d,cutoff):
    root=float(jn_zeros(l,n)[n-1])
    return jn(l,root*d/cutoff)*sph_harm(0,l,np.array(alpha),0).real*np.sqrt(2/cutoff**3/jn(l+1,root)**2)

def a_RBF(n,d,cutoff):
    return np.sqrt(2/cutoff)*np.sin(n*np.pi*d/cutoff)/d

def get_Kpoints_random(q,lattice,volume):
    a0=lattice[0,:]
    a1=lattice[1,:]
    a2=lattice[2,:]
    unit=2*np.pi*np.vstack((np.cross(a1,a2),np.cross(a2,a0),np.cross(a0,a1)))/volume
    ur=[(2*r-q-1)/2/q for r in range(1,q+1)]
    points=[]
    for i in ur:
        for j in ur:
            for k in ur:
                points.append(unit[0,:]*i+unit[1,:]*j+unit[2,:]*k)
    points=np.array(points) 
    return points  


def Phi(r,cutoff):
    return 1-6*(r/cutoff)**5+15*(r/cutoff)**4-10*(r/cutoff)**3
def gaussian(r,miuk,betak):
    return np.exp(-betak*(np.exp(-r)-miuk)**2)
def miuk(n,K,cutoff):
    # n=[1,K]
    return np.exp(-cutoff)+(1-np.exp(-cutoff))/K*n
def betak(K,cutoff):
    return (2/K*(1-np.exp(-cutoff)))**(-2)