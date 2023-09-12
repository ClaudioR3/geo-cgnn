[![DOI](https://zenodo.org/badge/688488915.svg)](https://zenodo.org/badge/latestdoi/688488915)


The key of the artificial intelligence is the data. Without it we are lost. 
When it's missing, we can consider external database. 
The worldwide crystal structure databases are AFLOW, COD, TCOD, Materials Project (MP), Materials Cloud NOMAD, odbx, Open Materials Database (omdb), and OQMD. 
It has been implemented a **Materials Project (MP) data extractor** that directly uses MP REST API to download the material data from MP database. 
This was useful for training the machine learning models and predicting a material property (in this case _formation energy_). 

# Quick start

**Download Materials Project dataset** (CIF, formation energy)

```bash
DATASET_PATH=<download path>
python ./data/mp_data.py $DATASET_PATH
```

**Training on MP dataset**

```bash
# change work directory
cd <path-to>/geo-cgnn/

# run training
python main.py --clean-torch --storage file --batch-size 64 \
  --lr 0.001 --optim Adam --train-ratio 0.7 --val-ratio 0.2 \
  --test-ratio 0.1 --workers 20 --epochs 300 --print-freq 1 \
  $DATASET_PATH
```

> **NOTE:**  It's recommended to use a GPU-supported machine.

**Inference on example dataset** 

```bash
cd <path-to>/geo-cgnn/
cp $DATASET_PATH/config_onehot.json ./data/example # copy config file into dataset path
python predict.py --modelpath experiments/model_best.pth.tar \
  --workers 10 --print-freq 1 ./data/example # run inference
```

Expected output:

```
-----------------------
INFO:    Could not find any nv binaries on this host!
=> loading model params 'experiments/model_best.pth.tar'
=> loaded model params 'experiments/model_best.pth.tar'
/opt/conda/lib/python3.7/site-packages/pymatgen/io/cif.py:1165: UserWarning: Issues encountered while parsing CIF: Some fractional co-ordinates rounded to ideal values to avoid issues with finite precision.
  warnings.warn("Issues encountered while parsing CIF: %s" % "\n".join(self.warnings))
=> loading model 'experiments/model_best.pth.tar'
=> loaded model 'experiments/model_best.pth.tar' (epoch 225, validation 0.04262223094701767)
---------Evaluate Model on Dataset---------------
Test: [1/1]     Time 8.793 (8.793)      Loss 0.0087 (0.0087)    MAE 0.094 (0.094)
 ** MAE 0.094

```

The predictions are saved in `<path-to>/geo-cgnn/output_pred`.

**Well done!**

## How to train
The learning (training) process of a neural network is an iterative process in which the calculations are carried out forward and backward through each layer in the network until the loss function is minimized.

The GEO-CGNN needs three types of files in the same directory for training:

  1. __cifs__: all CIF files those recode the crystal structures in <ID>.cif format, where ID is the unique ID for the crystal

  2. __label__: a CSV file with two columns. The first column recodes a unique ID for each crystal, and the second column recodes the value of target property.

  3. __config__: a JSON file recodes the number of atomic element using the onehot standard.


### **Run training phase:**

_List of params_:
```
python main.py -h
usage: main.py [-h] [--task {regression,classification}]
               [--storage {file,ceph}] [--disable-cuda] [-j N] [--epochs N]
               [--start-epoch N] [-b N] [--lr LR] [--lr-milestones N [N ...]]
               [--momentum M] [--weight-decay W] [--print-freq N]
               [--resume PATH] [--train-ratio N | --train-size N]
               [--val-ratio N | --val-size N] [--test-ratio N | --test-size N]
               [--optim SGD] [--disable-save-torch] [--clean-torch]
               OPTIONS [OPTIONS ...]

Geometric Information Enhanced Crystal Graph Network

positional arguments:
  OPTIONS               dataset options, started with the path to root dir,
                        then other options

optional arguments:
  -h, --help            show this help message and exit
  --task {regression,classification}
                        complete a regression or classification task (default:
                        regression)
  --storage {file,ceph}
                        Using file or ceph storage (default: file)
  --disable-cuda        Disable CUDA
  -j N, --workers N     number of data loading workers (default: 0)
  --epochs N            number of total epochs to run (default: 30)
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256)
  --lr LR, --learning-rate LR
                        initial learning rate (default: 0.01)
  --lr-milestones N [N ...]
                        milestones for scheduler (default: [100])
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 0)
  --print-freq N, -p N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  --train-ratio N       number of training data to be loaded (default none)
  --train-size N        number of training data to be loaded (default none)
  --val-ratio N         percentage of validation data to be loaded (default
                        0.1)
  --val-size N          number of validation data to be loaded (default 1000)
  --test-ratio N        percentage of test data to be loaded (default 0.1)
  --test-size N         number of test data to be loaded (default 1000)
  --optim SGD           choose an optimizer, SGD or Adam, (default: SGD)
  --disable-save-torch  Do not save CIF PyTorch data as .pkl files
  --clean-torch         Clean CIF PyTorch data .pkl files
```

_Running command example_:
~~~bash
python main.py \
 --storage file --batch-size 32  --lr 0.001 --optim SGD \
 --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1 \
 --workers 20 --epochs 100 --print-freq 1 \
 <path> 
~~~

> **NOTE:**  A computer with GPU support is required.


## How to test

Load the cif, the configuration and label files in the same directory.

> If you want to predict the material property only, create the label file where the first column is the cif ID and the second column (true material property) is 0 (zero) as below:
>
>   ```
>   0,0
>   2187,0
>   43,0
>   9,0
>   6561,0
>   19683,0
>   729,0
>   177147,0
>   27,0
>   81,0
>   ...
>   ```

### **Run the test phase:**

_List of params_:

```
    python predict.py -h
    usage: predict.py [-h] [--modelpath MODELPATH] [--storage {file,ceph}] [-j N]
                      [--disable-cuda] [--print-freq N] [--disable-save-torch]
                      [--train-val-test] [--only-pred]
                      cifpath

    Geometric Information Enhanced Crystal Graph Network
    
    positional arguments:
      cifpath               path to the directory of CIF files.
    
    optional arguments:
      -h, --help            show this help message and exit
      --modelpath MODELPATH
                            path to the trained model.
      --storage {file,ceph}
                            Using file or ceph storage (default: file)
      -j N, --workers N     number of data loading workers (default: 0)
      --disable-cuda        Disable CUDA
      --print-freq N, -p N  print frequency (default: 10)
      --disable-save-torch  Do not save CIF PyTorch data as .pkl; files
      --train-val-test      Return training/validation/testing results
      --only-pred           Jump Loss and MAE calcs
```

_Execution command example_:

~~~bash
    python predict.py \
    --print-freq 1 \
    --modelpath experiments/model_best.pth.tar \
    --workers 4 \
     <path>
~~~

The output will be like:
    
~~~bash
    => loading model 'experiments/model_best.pth.tar'
    => loaded model 'experiments/model_best.pth.tar' (epoch 92, validation 0.04700956493616104)
    /opt/conda/lib/python3.7/site-packages/pymatgen/io/cif.py:1165: UserWarning: Issues encountered while     parsing CIF: Some fractional co-ordinates rounded to ideal values to avoid issues with finite precision.  
    warnings.warn("Issues encountered while parsing CIF: %s" % "\n".join(self.warnings))
    ---------Evaluate Model on Dataset---------------
        Test: [1/2]     Time 17.663 (17.663)    Loss 5.6238 (5.6238)    MAE 2.099 (2.099)
        Test: [2/2]     Time 0.017 (8.840)      Loss 5.7916 (5.6606)    MAE 2.129 (2.106)
     ** MAE 2.106
~~~

The predictions will be saved in the `output_pred/prediction.csv` file. 
The CSV output file has three columns which consist of the CIF ID, the real material property (0 if you don't have it) and the predicted material property.

# Reference

This repository is based on: https://github.com/Tinystormjojo/geo-CGNN

# Citation

```
@misc{cheng21geo-cgnn,
    title={A geometric-information-enhanced crystal graph network for predicting properties of materials},
    author={Cheng, J., Zhang, C. & Dong, L. },
    year={2021},
    archivePrefix={www.nature.com},
    doi=https://doi.org/10.1038/s43246-021-00194-3
}
```

```
@misc{Ronchetti22,
    title={Machine learning techniques for data analysis in materials science},
    author={Ronchetti C., Puccini M., Ferlito S., Giusepponi S., Palombi F., Buonocore F., Celino M.},
    year={2022},
    archivePrefix={https://ieeexplore.ieee.org/},
    doi=10.23919/AEIT56783.2022.9951839
}
```
