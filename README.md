# Hilbert Stochastic Interpolants

## Setup
Install Python dependencies using `conda`

```
conda create --prefix="./.conda" python=3.12
conda activate ./.conda
pip install -r requirements.txt


```

## Downloading Datasets

To generate the  dataset for Darcy flow in 1D (`darcy_1d`):
```
python run.py dataset-gen darcy_1d
```
To download the  dataset for Darcy flow in 2D (`darcy_2d`):

```
huggingface-cli download  itoscholes/darcy --repo-type dataset --local-dir ./data/darcy --local-dir-use-symlinks False
```
To download the dataset for Navier-Stokes (`ns_nonbounded`):
```
huggingface-cli download  itoscholes/ns-nonbounded --repo-type dataset --local-dir ./data/ns-nonbounded --local-dir-use-symlinks False
```

## Training the model

To train the model:

```
python run.py train --config=./path/to/config.yml --n-save-every=<..? epochs>
```

Snapshots are saved to `./out/`

To calculate relative l2 error on the test set for both forward and inverse tasks:

```
python run.py test --config=./path/to/config.yml --pth=./path/to/snapshot.pth --stats-out=./path/to/stats.csv --out-file=./path/to/samples.npz
```

More options are available. Run `python run.py --help` or `python run.py train --help`, etc., to view these extra options
