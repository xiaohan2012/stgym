import pandas as pd

from stgym.data_loader import (
    BRCADataset,
    HumanCRCDataset,
    HumanIntestineDataset,
    MousePreopticDataset,
    MouseSpleenDataset,
)
from stgym.data_loader import get_info as get_ds_info

ds_list = [
    BRCADataset(root="data/brca"),
    HumanCRCDataset(root="data/human-crc"),
    HumanIntestineDataset(root="data/human-intestine"),
    MousePreopticDataset(root="data/mouse-preoptic"),
    MouseSpleenDataset(root="data/mouse-spleen"),
]


def get_info(ds):
    name = ds.root.split("/")[1]
    ds_info = get_ds_info(name)
    return {
        "name": name,
        "size": len(ds),
        "n_feats": ds.num_features,
        "n_classes": len(ds.y.unique()),
        "task_type": ds_info["task_type"],
        "data_source_url": ds_info["data_source_url"],
        "used_in_paper": ds_info["used_in_paper"],
    }


ds_df = pd.DataFrame([get_info(ds) for ds in ds_list])
print(ds_df.to_markdown(index=None))
