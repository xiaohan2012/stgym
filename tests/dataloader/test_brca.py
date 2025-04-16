from stgym.data_loader.brca import BRCADataset

def test():
    ds = BRCADataset(root='./data/brca')
    assert len(ds) == 347

