from torch_geometric.data import InMemoryDataset  # , download_url


class AbstractDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        graph_construction_tag: str | None = None,
    ):
        # Must be set before super().__init__() — PyG calls processed_file_names during init
        self._graph_construction_tag = graph_construction_tag
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        raise NotImplementedError

    @property
    def processed_file_names(self):
        if self._graph_construction_tag:
            return [f"data_{self._graph_construction_tag}.pt"]
        return ["data.pt"]

    def download(self):
        # Download to `self.raw_dir`.
        # download_url(url, self.raw_dir)
        # raise NotImplementedError("Download not supported yet!")
        pass

    def process_data(self):
        raise NotImplementedError

    def process(self):
        data_list = self.process_data()

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
        return data_list
