# use
# python get_pos_maxpsan.py <dataset_name>
# to obtain the max_span info
__data__ = {
    "brca": {
        "min_span": 297.6280212402344,
        "max_span": 1018.4807739257812,
        "num_classes": 2,
    },
    "human-crc": {"min_span": 795.0, "max_span": 1919.0, "num_classes": 10},
    "mouse-spleen": {"min_span": 0.0, "max_span": 1341.0, "num_classes": 58},
    "mouse-preoptic": {
        "min_span": 6257.4140625,
        "max_span": 9512.359375,
        "num_classes": 6,
    },
    "human-intestine": {"min_span": 9070.0, "max_span": 9406.0, "num_classes": 21},
}


def get_info(ds_name: str) -> dict:
    return __data__[ds_name]
