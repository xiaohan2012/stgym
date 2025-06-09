__data__ = {"brca": {"max_span": 731.96027}, "human-crc": {"max_span": 0.0}}


def get_info(ds_name: str) -> dict:
    return __data__[ds_name]
