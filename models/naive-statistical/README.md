# Naive statistical model

## Dependencies

Use the packet manager [pip](https://pip.pypa.io/en/stable/) to install the relevant dependencies.

## Usage

In the jupyter notebook in In [2] specify the correct paths to the corresponding data:

* Set *DIR_SRC* to the path corresponding to the directory with the labeled cosmology images.
* Set *DIR_BIN* for temporarily saving the binary images used later for the hough transform.
* Set *DIR_TAR* as target directory for storing the generated cosmology images.

```bash
DIR_SRC = "cil-cosmology-2020/labeled"
DIR_BIN = "cil-cosmology-2020/binary"
DIR_TAR = "cil-cosmology-2020/results"
label_csv = pd.read_csv("cil-cosmology-2020/labeled.csv")
```
