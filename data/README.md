
# Materials Project Data Extractor

The `data/mp_data.py` python script extracts data from Materials Project Database (MP, [link](https://next-gen.materialsproject.org/))
and it saves the data as cif files in the indicated path. 

The mp_data.py script has following command line interface:

~~~bash
usage: mp_data.py [-h] [--json-data JSON_DATA] [--json-filter JSON_FILTER] [--storage {file,ceph}]
                  [--ceph-auth CEPH_AUTH] [--no-download] [--no-stats]
                  cif-dir

Materials Project Data Extractor

positional arguments:
  cif-dir               The directory to download the cif files

optional arguments:
  -h, --help            show this help message and exit
  --json-data JSON_DATA
                        Path of the data json file
  --json-filter JSON_FILTER
                        Path of the filter json file
  --storage {file,ceph}
                        File or Ceph storage type
  --ceph-auth CEPH_AUTH
                        Path of the Ceph authentication properties
  --no-download         Jump download phase
  --no-stats            Jump statistics phase
~~~

Run the script using `python mp_data.py` command.

## Download
The download process requires to save the temporaly MP data in the default (`data.json` ) or a custom (using _--json-data_ option) json file (~250MB of size). If you have already downloaded the data and don't want to repeat the download step, use _--no-download_ option.
Furthermore, the user can apply a filter to download a subset of MP data by editing the `filter.json` json file. If the file is empty, no criteria is applied. 
For example:

- Use _task_id_ to download a material where task_id key is in fact the materials id for the Materials Project

  ~~~json
  {"task_id": "mp-1234"}
  ~~~

- or search for the material using the _pretty_formula_ key

  ~~~json
  {"pretty_formula": "Li2O"}
  ~~~

  For a more complicated example you can identify the appropriate key by going to the materials/spacegroup/symbol subfolder in the [github reposiory](https://github.com/materialsproject/mapidoc).
Some interesting filters are:

- Select structures with more than or equal to 2 unique elements

  ~~~json
  {"nelements":{"$gte": 2}}
  ~~~

- Select structures if they have unique elements in the list ["Na", "Mn", "O"] 

  ~~~json
  {"elements":{"$in" : ["Na", "O", "Mn"]}}
  ~~~

- Select structures if they have __not__ unique elements in the list ["He", "Ne", "Ar", "Kr", "Xe"] 

  ~~~json
  {"elements":{"$nin" : ["He", "Ne", "Ar", "Kr", "Xe"]}}
  ~~~

See [Materials Project API](https://next-gen.materialsproject.org/api) for more details.

## Statistics
The second phase aplies three statistics to show to the user the main features of the downloaded dataset and it saves them into `data/stats`. The statistics are:

  1. Count the structures by the number of elements

  2. Count elements in the structure visualized by periodic table

  3. Mean formation energy for each element

These statistics can be jumped by using _--no-stats_ option. Following the statistics using the filter:

~~~json
{"elements":{"$nin" : ["He", "Ne", "Ar", "Kr", "Xe"]}}
~~~

### Count unique elements species in the structures

The following image shows that most structures have two to five unique elements.

![count](https://github.com/ClaudioR3/geo-cgnn/assets/18485450/411abb19-af0a-4615-90e9-8030dd595fea)


### Count elements in the structure visualized by periodic table

Here, the following image shows that the most present element in the MP dataset is oxygen (O) with more than 60,000 structures while the least present elements are actinium (Ac) and protactinium (Pa) with approximately 300 structures. The gray-colored elements aren't present in the MP dataset. See [countByElement.csv](uploads/d51627725bee4a5095a781344a56cc6a/countByElement.csv) for details.

![fig3](https://github.com/ClaudioR3/geo-cgnn/assets/18485450/c7b9e35c-ce6d-4db3-9da6-d29e4cfeed21)

### Mean formation energy for each element

The following image shows that the element with the highest average formation energy (afe) is technetium (Tc) with more than 0.5 eV/atom while the element with the lowest afe is fluorine (F) with less than -2.5 eV/atom. The gray-colored elements aren't present in the MP dataset. See [meanFormEnergyByElement.csv](uploads/55c97aa70677627412da243feec39f3f/meanFormEnergyByElement.csv) for details.
The most interesting **transaction metals** are vanadium (V) and titanium (Ti) with less than -2.0 eV/atom.

![fig4](https://github.com/ClaudioR3/geo-cgnn/assets/18485450/4c0814c3-5811-461c-9786-e9952bad7651)
