from stgym.data_loader import get_dataset_class
from stgym.utils import get_coord_span


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Get maximum coordinate span for a dataset"
    )
    parser.add_argument("dataset_name", type=str, help="Name of the dataset")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    ds_cls = get_dataset_class(dataset_name)
    ds = ds_cls(root=f"./data/{dataset_name}")
    print(get_coord_span(ds))


if __name__ == "__main__":
    main()
