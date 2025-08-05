# use
# python get_pos_maxpsan.py <dataset_name>
# to obtain the max_span info
__data__ = {
    "brca": {
        "min_span": 297.6280212402344,
        "max_span": 1018.4807739257812,
        "num_classes": 2,
        "task_type": "graph-classification",
        "data_source_url": "https://zenodo.org/records/6376767",
        "used_in_paper": "Unsupervised discovery of tissue architecture in multiplexed imaging",
    },
    "human-crc": {
        "min_span": 795.0,
        "max_span": 1919.0,
        "num_classes": 10,
        "task_type": "node-clustering",
        "data_source_url": "https://data.mendeley.com/datasets/mpjzbtfgfr/1",
        "used_in_paper": "Unsupervised and supervised discovery of tissue cellular neighborhoods from cell phenotypes.",
    },
    "mouse-spleen": {
        "min_span": 0.0,
        "max_span": 1341.0,
        "num_classes": 58,
        "task_type": "node-classification",
        "data_source_url": "https://data.mendeley.com/datasets/zjnpwh8m5b/1",
        "used_in_paper": "Unsupervised and supervised discovery of tissue cellular neighborhoods from cell phenotypes.",
    },
    "mouse-preoptic": {
        "min_span": 6257.4140625,
        "max_span": 9512.359375,
        "num_classes": 6,
        "task_type": "graph-classification",
        "data_source_url": "https://datadryad.org/dataset/doi:10.5061/dryad.8t8s248",
        "used_in_paper": "Unsupervised and supervised discovery of tissue cellular neighborhoods from cell phenotypes.",
    },
    "human-intestine": {
        # has only 8 data points
        "min_span": 9070.0,
        "max_span": 9406.0,
        "num_classes": 21,
        "task_type": "node-classificatioin",
        "data_source_url": "https://datadryad.org/landing/show?id=doi%3A10.5061%2Fdryad.g4f4qrfrc",
        "used_in_paper": "Annotation of spatially resolved single-cell data with STELLAR",
    },
    "human-lung": {
        "min_span": 13.316166877746582,
        "max_span": 37.27265167236328,
        "num_classes": 48,  # there are a few cell types with very few occurrences
        "task_type": "node-classificatioin",
        "data_source_url": "https://www.lungmap.net/dataset/?experiment_id=LMEX0000004396&view=downloads",
        "used_in_paper": "The human body at cellular resolution: the NIH Human Biomolecular Atlas Program",
    },
}


def get_info(ds_name: str) -> dict:
    return __data__[ds_name]
