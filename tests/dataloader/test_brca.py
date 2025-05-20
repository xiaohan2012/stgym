from stgym.data_loader.brca import BRCADataset


def test():
    # TODO: use test data and do a proper test
    ds = BRCADataset(root="./data/brca")
    assert len(ds) == 347
