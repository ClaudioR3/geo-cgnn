#!/usr/bin/env python
# coding: utf-8

import logging
import json, requests
import IProgress
from pymatgen.ext.matproj import MPRester
from pprint import pprint
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse, sys
import utils as Utils
from tqdm import tqdm
from pymatgen.core.structure import Structure
## Logging
loglevel = 'INFO'
level = logging.getLevelName(loglevel)
logging.basicConfig(format='%(levelname)s: %(message)s', level=level)
log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='AI4MAT - Materials Project Data Extractor')
parser.add_argument('cif_dir', help='The directory to download the cif files')
parser.add_argument('--json-data', default='./data.json', type=str, help='Path of the data json file')
parser.add_argument('--json-filter', default='./filter.json', type=str, help='Path of the filter json file')
parser.add_argument('--storage', default='file', type=str, help='File or Ceph storage type', choices=['file', 'ceph'])
parser.add_argument('--ceph-auth', default='./ceph.properties', type=str, help='Path of the Ceph authentication properties')
parser.add_argument('--no-download', action='store_true', help='Jump download phase')
parser.add_argument('--no-stats', action='store_true', help='Jump statistics phase')

args = parser.parse_args(sys.argv[1:])

def main():
    global args

    if not args.no_download:
        API_KEY = "BrCAlfETWO2GXyVJ"
        mpr = MPRester(api_key= API_KEY)
        log.info("Material Project API Rester created")
        with open(args.json_filter,'r') as f:
            criteria = json.load(f)
        log.info("Downloading data with filter: \n{}".format(criteria))
        data = mpr.query(
            criteria = criteria,
            properties = [
                'material_id', 
                'elements',
                'nelements',
                'pretty_formula',
                'energy', 
                'formation_energy_per_atom',
                'cif'
            ]
        )
        log.info("Recieved {} materials!".format(len(data)))

        log.info("Saving the materials in json file: {} ...")
        with open(args.json_data, 'w') as f:
            json.dump(data, f)
        log.info("Saved!")
    else:
        with open(args.json_data, 'r') as f:
            data=json.load(f)
        log.warning("The Material Project data download was jumped.")

    # # ***Statistics***
    # 
    #     Here, a statistic series.

    if not args.no_stats:

        #     1. Number of elements
        log.info("1. Number of elements")
        
        noe_list = [x['nelements'] for x in data]
        noe = {k: noe_list.count(k) for k in sorted(set(noe_list))}

        # plot
        fig, ax = plt.subplots()
        w = 0.75
        x = np.arange(len(noe.keys())) + 1
        y = noe.values()
        ax.bar(x , y)

        ax.set_xticks(x )
        ax.set_xticklabels(map(str, x ))
        ax.set_yscale('log')

        ax.set_xlabel('Number of elements')
        ax.set_ylabel('Count')

        plt.show()


        #     2. Count elements in the structure visualized by periodic table
        
        log.info("2. Count elements in the structure visualized by periodic table")
        
        loe = []
        for x in data:
            loe+=x['elements']
        elems = set(loe)
        coe = {k: loe.count(k) for k in elems}

        strBuild = ''
        for k,v in coe.items():
            strBuild+= str(k) + ',' + str(v) + '\n'

        with open('stats/countByElement.csv', 'w') as f:
            f.write(strBuild)

        os.system('python utils/ptable.py stats/countByElement.csv --log_scale 1 ')


        #     3. Mean formation energy for each element
        
        log.info("3. Mean formation energy for each element")

        mfe = {k: [x['formation_energy_per_atom'] for x in data if k in x['elements']] for k in elems}

        strBuild = ''
        for k,v in mfe.items():
            strBuild+= str(k) + ',' + str(sum(v)/len(v)) + '\n'

        with open('stats/meanFormEnergyByElement.csv', 'w') as f:
            f.write(strBuild)

        os.system('python utils/ptable.py stats/meanFormEnergyByElement.csv # --log_scale 1')


    # # ***CIFData dataset***:
    # 
    #     The CIFData dataset is a wrapper for a dataset where the crystal structures
    #     are stored in the form of CIF files. The dataset should have the following
    #     directory structure:
    # 
    #     root_dir
    #     ├── id_prop.csv
    #     ├── atom_init.json
    #     ├── id0.cif
    #     ├── id1.cif
    #     ├── ...
    # 
    #     id_prop.csv: a CSV file with two columns. The first column recodes a
    #     unique ID for each crystal, and the second column recodes the value of
    #     target property.
    # 
    #     atom_init.json: a JSON file that stores the initialization vector for each
    #     element.
    # 
    #     ID.cif: a CIF file that recodes the crystal structure, where ID is the
    #     unique ID for the crystal.


    # ***Relevant filenames***

    json_file = args.json_data # path to dictionary of properties for all materials
    
    if args.storage=='file':
        storage=Utils.FileStorage(args.cif_dir)
    elif args.storage=='ceph':
        storage=Utils.CephStorage(args.cif_dir, args.ceph_auth)


    # ---------------------------
    # ### ***ID_PROP.csv***
    #     a CSV file with two columns. The first column recodes a
    #     unique ID for each crystal, and the second column recodes the value of
    #     target property.
    # Until now, we need that energies to be the label/properties. So opt-energies.csv is the id_prop.csv
    # optional properties: 'energy','formation_energy_per_atom', 'band_gap'
    log.info("Creating id_prop.csv in {} using {} storage type".format(args.cif_dir, args.storage))
    s = ""
    for i, x in enumerate(data):
        s+=str(i)+","+str(x['formation_energy_per_atom'])+"\n"

    #with open(os.path.join(destination_dir, "id_prop.csv"), "w") as f:
    #    f.write(s)
        
    storage.write(os.path.join(args.cif_dir, "id_prop.csv"), s.encode('utf-8'))

    #os.system('head  {destination_dir}id_prop.csv')


    # ---------------------------
    # ### ***ATOM_INIT.json***: 
    #     a JSON file that stores the initialization vector for each
    #     element.

    log.info("Creating atom_init.json in {} using {} storage type".format(args.cif_dir, args.storage))

    elements = []
    for x in data:
        elements += x['elements']
    elements = sorted(set(elements))

    def init_vector(symbols):
        s = set(symbols)
        return [1 if elem in s else 0 for elem in elements]

    init_vector = {
        i: init_vector(x['elements']) 
        for i,x in enumerate(data)
    }

    #with open(os.path.join(destination_dir,"atom_init.json"), 'w') as f: 
    #    f.write( json.dumps( init_vector ) )
    storage.write(os.path.join(args.cif_dir, "atom_init.json"), json.dumps(init_vector).encode('utf-8'))

    #get_ipython().system('head -c 1000 {destination_dir}atom_init.json')
    
    # ---------------------------
    # ### ***config_onehot.json*** 
    #     a json file that recodes the onehot of elements.

    log.info("Creating onehot file saved as config_onehot.json in {} using {} storage type".format(args.cif_dir, args.storage))
    def build_config(cifs):
        atoms=[]
        
        for cif in tqdm(cifs):
            crystal = Structure.from_str(
                cif,
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
        return config
    
    config=build_config([x['cif'] for x in data])
    
    storage.write(os.path.join(args.cif_dir, 'config_onehot.json'), body=json.dumps(config).encode('utf-8'))


    # ---------------------------
    # ### ***ID.cif*** 
    #     a CIF file that recodes the crystal structure, where ID is the
    #     unique ID for the crystal.

    log.info("Creating cif files saved as <id>.cif in {} using {} storage type".format(args.cif_dir, args.storage))
    
    for i, x in enumerate(tqdm(data)):
        #with open(os.path.join(destination_dir, str(i)+'.cif'), 'w') as f: 
        #          f.write(x['cif'])
        
        storage.write(os.path.join(args.cif_dir, str(i)+'.cif'), x['cif'].encode('utf-8'))

    #os.system("cat {os.path.join(destination_dir, '0.cif')}")
    log.info("The cif files are loaded successfully in {}".format(args.cif_dir))
    

if __name__=='__main__':
    main()
