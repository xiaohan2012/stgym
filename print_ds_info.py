import pandas as pd

from stgym.data_loader import get_all_ds_names, get_dataset_class
from stgym.data_loader import get_info as get_ds_info

ds_list = [
    get_dataset_class(ds_name)(root=f"data/{ds_name}") for ds_name in get_all_ds_names()
]

for ds in ds_list:
    print(ds)
    for dt in ds:
        print(dt.x.device)
        break


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


cols = ["name", "size", "n_feats", "n_classes", "task_type"]
ds_df = pd.DataFrame([get_info(ds) for ds in ds_list])
print(ds_df.sort_values(by="task_type")[cols].to_markdown(index=None))
