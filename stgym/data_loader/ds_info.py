# use
# python get_pos_maxpsan.py <dataset_name>
# to obtain the max_span info
from .const import DatasetName

__data__ = {
    DatasetName.brca: {
        "min_span": 297.6280212402344,
        "max_span": 1018.4807739257812,
        "num_classes": 2,
        "task_type": "graph-classification",
        "data_source_url": "https://zenodo.org/records/6376767",
        "used_in_paper": "Unsupervised discovery of tissue architecture in multiplexed imaging",
    },
    DatasetName.human_crc: {
        "min_span": 795.0,
        "max_span": 1919.0,
        "num_classes": 10,
        "task_type": "node-clustering",
        "data_source_url": "https://data.mendeley.com/datasets/mpjzbtfgfr/1",
        "used_in_paper": "Unsupervised and supervised discovery of tissue cellular neighborhoods from cell phenotypes.",
    },
    DatasetName.mouse_spleen: {
        "min_span": 0.0,
        "max_span": 1341.0,
        "num_classes": 58,
        "task_type": "node-classification",
        "data_source_url": "https://data.mendeley.com/datasets/zjnpwh8m5b/1",
        "used_in_paper": "Unsupervised and supervised discovery of tissue cellular neighborhoods from cell phenotypes.",
    },
    DatasetName.mouse_preoptic: {
        "min_span": 6257.4140625,
        "max_span": 9512.359375,
        "num_classes": 6,
        "task_type": "graph-classification",
        "data_source_url": "https://datadryad.org/dataset/doi:10.5061/dryad.8t8s248",
        "used_in_paper": "Unsupervised and supervised discovery of tissue cellular neighborhoods from cell phenotypes.",
    },
    DatasetName.human_intestine: {
        # has only 8 data points
        "min_span": 9070.0,
        "max_span": 9406.0,
        "num_classes": 21,
        "task_type": "node-classificatioin",
        "data_source_url": "https://datadryad.org/landing/show?id=doi%3A10.5061%2Fdryad.g4f4qrfrc",
        "used_in_paper": "Annotation of spatially resolved single-cell data with STELLAR",
    },
    DatasetName.human_lung: {
        "min_span": 13.316166877746582,
        "max_span": 37.27265167236328,
        "num_classes": 48,  # there are a few cell types with very few occurrences
        "task_type": "node-classificatioin",
        "data_source_url": "https://www.lungmap.net/dataset/?experiment_id=LMEX0000004396&view=downloads",
        "used_in_paper": "The human body at cellular resolution: the NIH Human Biomolecular Atlas Program",
    },
    DatasetName.breast_cancer: {
        "min_span": 22.807723999023438,
        "max_span": 31.39260482788086,
        "num_classes": 39,
        "task_type": "node-classification",
        "data_source_url": "https://cellxgene.cziscience.com/collections/4195ab4c-20bd-4cd3-8b3d-65601277e731",
        "used_in_paper": "CellContrast: Reconstructing spatial relationships in single-cell RNA sequencing data via deep contrastive learning",
    },
    DatasetName.mouse_kidney: {
        "min_span": 4708.080078125,
        "max_span": 5432.63720703125,
        "num_classes": 3,
        "task_type": "graph-classification",
        "data_source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE190094",
        "used_in_paper": "High-resolution Slide-seqV2 spatial transcriptomics enables discovery of disease-specific cell neighborhoods and pathways",
    },
    DatasetName.cellcontrast_breast: {
        "min_span": 16.34262733088118,
        "max_span": 16.953102237662257,
        "num_classes": 8,
        "task_type": "node-classification",
        "data_source_url": "https://cellxgene.cziscience.com/collections/4195ab4c-20bd-4cd3-8b3d-65601277e731",
        "used_in_paper": "CellContrast: Reconstructing spatial relationships in single-cell RNA sequencing data via deep contrastive learning",
    },
    DatasetName.colorectal_cancer: {
        "min_span": 23.90913963317871,
        "max_span": 25.46361541748047,
        "num_classes": 38,
        "task_type": "node-classification",
        "data_source_url": "https://zenodo.org/records/15042463",
        "used_in_paper": "High-definition spatial transcriptomic profiling of immune cell populations in colorectal cancer",
    },
}


def get_all_ds_names() -> list[str]:
    return list(__data__.keys())


def get_info(ds_name: str) -> dict:
    return __data__[ds_name]
