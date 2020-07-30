# cil-cosmology-2020

## Plots

The script for the histogram plot can be found in `tools/histograms.py`, the one to compare model scores in `tools/score-comp.py`

## Kaggle Competition

In order to reproduce our results from kaggle, do the following:
```
cd models/regression3
python3 train.py --data_dir=[DATA_DIR]
```

This will write the `pred.csv` file into the given data directory.

## Image Generation

Use the following instructions to reproduce the results reported in our paper.

### Naive Model

In the jupyter notebook in `models/naive-statistical` specify the correct paths to the corresponding data:

* Set *DIR_SRC* to the path corresponding to the directory with the labeled cosmology images.
* Set *DIR_BIN* for temporarily saving the binary images used later for the hough transform.
* Set *DIR_TAR* as target directory for storing the generated cosmology images.

```
DIR_SRC = "cil-cosmology-2020/labeled"
DIR_BIN = "cil-cosmology-2020/binary"
DIR_TAR = "cil-cosmology-2020/results"
label_csv = pd.read_csv("cil-cosmology-2020/labeled.csv")
```

### StyleGAN

```
cd models/stylegan1
pip3 install tensorflow-gpu==1.15 
python3 dataset_tool.py create_cosmology [OUTPUT_DIR] [DATA_DIR]
python3 train.py --data_dir [OUTPUT_DIR]
```

### Layered Model

```
cd models/layered-statistical
pip3 install torch torchvision
PYTHONPATH=../../ python3 prep_data.py --dataDir=[DATA_DIR]
cd coord_gen
python3 gan.py
cd ../image_gen
python3 gan.py
cd ..
python3 generate_galaxies.py
```
