import torch
from torch.utils.data import DataLoader
from pathlib import Path

import time

#from neuralop.data.datasets.hdf5_dataset import H5pyDataset
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.data.transforms.data_processors import DefaultDataProcessor

from .modified_hdf5 import H5pyDataset

class FullSizeNavierStokes(object):
    def __init__(self, root_dir, n_train=10000, n_test=2000, res=1024):
        
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
            
          
        self._train_db = H5pyDataset(root_dir / "nsforcing_1024_train.hdf5", resolution=res, n_samples=n_train)
        self._test_db = H5pyDataset(root_dir / "nsforcing_1024_test.hdf5", resolution=res, n_samples=n_test)

        t0 = time.time()
        print("Loading x_train...")
        # limit length to fit encoder for time
        x_train = self._train_db.data["x"][:1000]
        x_train = torch.tensor(x_train, dtype=torch.float32)
        if x_train.ndim == 3:
            x_train = x_train.unsqueeze(1)
        print(f"{x_train.shape=}")
        t1 = time.time()
        print(f"loaded x_train in {t1-t0} sec")
        channel_dim = 1
        # create input encoder
        reduce_dims = list(range(x_train.ndim))
        # preserve mean for each channel
        reduce_dims.pop(channel_dim)
        
        t2 = time.time()
        input_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        print("Fitting encoder for x_train...")
        input_encoder.fit(x_train)
        t3 = time.time()
        print(f"took {t3-t2} sec to fit encoder for x_train")
        del x_train

        # create output encoder
        print("Loading y_train...")
        y_train = self._train_db.data["y"][:1000]
        y_train = torch.tensor(y_train, dtype=torch.float32)
        if y_train.ndim == 3:
            y_train = y_train.unsqueeze(1)
        reduce_dims = list(range(y_train.ndim))
        # preserve mean for each channel
        reduce_dims.pop(channel_dim)

        output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        print("Fitting encoder for y_train...")
        output_encoder.fit(y_train)
        del y_train

        self.data_processor = DefaultDataProcessor(in_normalizer=input_encoder,
                                                   out_normalizer=output_encoder)
    @property
    def train_db(self):
        return self._train_db

    @property
    def test_db(self):
        return self._test_db
    
if __name__ == "__main__":
    from neuralop.layers.embeddings import GridEmbedding2D
    pos_embed = GridEmbedding2D()
    ns_dataset = FullSizeNavierStokes(root_dir=Path("/home/dave/data/navier_stokes/ns1024_full"), res=1024)
    print(f"{len(ns_dataset.test_db)=}")
    train_loader = DataLoader(ns_dataset.test_db, batch_size=1)
    print(train_loader)
    dproc = ns_dataset.data_processor
    print(dproc) 
    for idx, batch in enumerate(train_loader):
        #print(batch)
        print(f"{torch.mean(batch['x'])=}")
        print(batch['x'].shape)
        print(batch['y'].shape)
        batch = dproc.preprocess(batch)
        print(f"{torch.mean(batch['x'])=}")
        print(pos_embed(batch['x']).shape)
        if idx ==1:
            raise ValueError()