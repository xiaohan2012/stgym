# Background

You will work on a data preprocessing task on a spatial transcriptomics dataset. Later, the processed dataset will be consumed by graph neural networks (GNN).

You will start with a CSV file, select the relevant columns, and write a Python wrapper class to process the data.

Finally, the processed dataset needs to be in the format supported by pytorch_geometric, since this library is used to implement the GNN models


# Input dataset format

The dataset is a data table, in which each row corresponds to a cell inside a sample (such as tissue).

Columns encode the following information:

- The spatial location of the cell (e.g., the X/Y coordinates)
- Identification of the sample
- Ground truth label of the cell (e.g., cell type)
- Gene expression / marker information, the cell-level 'features'. These columns often take up the majority of the table content
- Other miscellaneous information of the subject that the sample is taken from, such as sex, age, etc

# Task specification

## Step 0: copy the raw CSV file to project folder

- The CSV file path and dataset name will be specified in later instructions.
  - Ask me if they're not provided
- Data folder is at `{PROJECT_ROOT}/data/{dataset_name}`
  - If you're under a git worktree, make sure the data is copied to the main worktree

## Step 1: select the relevant columns

You need to identify the following columns which encode the following:

- Spatial location of the cell (2 columns)
- Sample-level identification (1 column)
- Ground truth label of the cell (1 column)
- Gene expression / marker information (multiple columns)


## Step 2: write the Python script using torch_geometric interface

The script contains the following:

- column-related variables
- The wrapper class to process the raw csv file

For example:

- `human_intestine` dataset:
   - raw file: `data/human-intestine/raw/source.csv`
   - Python script: `stgym/data_loader/human_intestine.py`
- `mouse_spleen` dataset:
   - raw file: `data/mouse-spleen/raw/source.csv`
   - Python script: `stgym/data_loader/mouse_spleen.py`

## Step 3: write unit tests

Refer to the unit tests under `./tests/dataloader/test_*.py`.

## Step 4: update other relevant files

1. add the new dataset name in `stgym/data_loader/const.py`
2. update `get_dataset_class` in `stgym/data_loader/__init__.py`
3. add the dataset info in `stgym/data_loader/ds_info.py`

Remarks:

- to obtain the `min_span` and `max_span` info for this new dataset, use the script get_pos_maxpsan.py
- to obtain `num_classes`, load the dataset to inspect

# Further information about the dataset

#$ARGUMENTS
