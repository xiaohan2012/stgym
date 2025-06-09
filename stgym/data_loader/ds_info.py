# use
# python get_pos_maxpsan.py <dataset_name>
# to obtain the max_span info
__data__ = {"brca": {"max_span": 1018.48}, "human-crc": {"max_span": 1919.00}}


def get_info(ds_name: str) -> dict:
    return __data__[ds_name]
