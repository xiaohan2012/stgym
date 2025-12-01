# STGym Data Loader Documentation

This directory contains data loaders for various spatial transcriptomics datasets used in STGym.

## Human Pancreas Development Dataset

This dataset contains spatial transcriptomics data from human pancreatic development at different post-conception weeks (PCW) for node classification tasks.

### Data Source

**Download the raw data from GEO repository:**
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE197317

1. Visit the link above and download all files from GSE197317_RAW
2. Extract all `.tar.gz` files to get 10x Visium spatial data directories
3. Keep the `.csv.gz` files containing cell type deconvolution data

### Data Preprocessing

Use the preprocessing script to convert raw 10X Visium data and cell type deconvolution into the consolidated dataset:

```bash
python scripts/data_preprocessing/process_human_pancreas.py \
    "/path/to/extracted/GSE197317_RAW" \
    --output data/human-pancreas/raw/source.csv
```

Top-1000 genes with highest standard deviation in activation values are included.

#### Parameters:
- `input_directory`: Path to the directory containing extracted GSE197317 data (both 10x directories and deconvolution CSV files)
- `--output`: Output CSV file path (default: `data/human-pancreas/raw/source.csv`)

#### What the script does:
1. Loads 10X Visium spatial coordinates and gene expression from all 8 samples
2. Loads cell type deconvolution data for each sample
3. Determines dominant cell type for each spatial spot as ground truth label
4. Selects top 1000 variable genes as features
5. Adds cell type proportions and spatial coordinates as additional features
6. Outputs consolidated `source.csv` for node classification

#### Generated files:
- `source.csv`: Main dataset with spatial spots as rows and features as columns


## Glioblastoma Dataset

This dataset contains spatial transcriptomics data from glioblastoma patients processed into gene expression features for graph classification tasks.

### Data Source

**Download the raw data from Dryad repository:**
https://datadryad.org/dataset/doi:10.5061/dryad.h70rxwdmj

1. Visit the link above (free account may be required)
2. Download the dataset
3. Extract to get the `10XVisium 2` directory containing 28 sample folders

### Data Preprocessing

Use the preprocessing script to convert raw 10X Visium data into the consolidated dataset:

```bash
python scripts/data_preprocessing/create_glioblastoma.py \
    --input-dir "/path/to/extracted/10XVisium 2" \
    --output-dir ./data/glioblastoma/raw \
    --top-genes 500
```

#### Parameters:
- `--input-dir`: Path to the extracted "10XVisium 2" directory
- `--output-dir`: Output directory (default: `./data/glioblastoma/raw`)
- `--top-genes`: Number of genes to select (default: 1000)

#### What the script does:
1. Loads 10X Visium filtered gene expression matrices from all 28 samples
2. Selects genes using biological relevance and expression variability
3. Applies log-transformation: `log(counts + 1)`
4. Extracts spatial coordinates and tissue type labels
5. Outputs consolidated `source.csv` with 500 gene expression features per spatial spot

#### Generated files:
- `source.csv`: Main dataset (88,793 spots × 500 genes)
- `gene_info.csv`: Selected gene metadata
- `dataset_summary.txt`: Processing statistics

### Binary Classification Relabeling

For improved classification performance, the dataset supports binary relabeling that consolidates tumor subtypes:

- **cortex**: Remains as "cortex" (8 samples, 23,819 spots)
- **tumor**: Combines "tumor", "tumor_core", and "tumor_infiltration" (20 samples total, 64,974 spots)

**Motivation**: The original tumor_core (1 sample) and tumor_infiltration (1 sample) classes have very small sample sizes that can lead to poor generalization in machine learning models. By consolidating these into a single "tumor" class, we create a more balanced binary classification task suitable for robust model training.

This relabeling is automatically applied in the GlioblastomaDataset loader when processing the data.
