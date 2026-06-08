from torch.utils.data import DataLoader
from musdb_dataset import MusdbDataset


class MusdbDataModule:
    def __init__(self, musdb_root: str, subset: str = 'train', sr: int = 16000, batch_size: int = 1, num_workers: int = 8, segment_length: float = None):
        self.musdb_root = musdb_root
        self.subset = subset
        self.sr = sr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.segment_length = segment_length

        self.train_dataset = None
        self.val_dataset = None

    def setup(self):
        # Expect directory layout: <musdb_root>/train, <musdb_root>/valid, <musdb_root>/test
        self.train_dataset = MusdbDataset(self.musdb_root, subset='train', sr=self.sr, segment_length=self.segment_length)
        # Validation/test should be full-length (no segmenting) so we set segment_length=None
        self.val_dataset = MusdbDataset(self.musdb_root, subset='valid', sr=self.sr, segment_length=None)

    def train_dataloader(self):
        if self.train_dataset is None:
            self.setup()
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        if self.val_dataset is None:
            self.setup()
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=True)
