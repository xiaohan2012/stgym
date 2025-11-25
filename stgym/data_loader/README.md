# STGym Data Loader Documentation

This directory contains data loaders for various spatial transcriptomics datasets used in STGym.

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

### Dataset Statistics

- **28 samples** from 20 glioblastoma patients
- **88,793 total spatial spots** across all samples
- **500 gene expression features** per spatial spot
- **4 tissue types**:
  - cortex: 8 samples, 23,819 spots
  - tumor: 18 samples, 59,652 spots
  - tumor_core: 1 sample, 1,853 spots
  - tumor_infiltration: 1 sample, 3,469 spots
