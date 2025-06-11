# use
# python get_pos_maxpsan.py <dataset_name>
# to obtain the max_span info
__data__ = {
    "brca": {"max_span": 1018.48, "num_classes": 2},
    "human-crc": {"max_span": 1919.00, "num_classes": 10},
    "mouse-spleen": {"max_span": 1341.0, "num_classes": 58},
    "mouse-preoptic": {"max_span": 9512.359375, "num_classes": 5},
}


def get_info(ds_name: str) -> dict:
    return __data__[ds_name]
