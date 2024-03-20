import numpy as np
import torch

from src.data_manager.data_manager import SequentialTrainDataset


class SequentialTrainDatasetWithAlbumCoversEmbeddings(SequentialTrainDataset):
    # This class is used to load the training set. If a playlist is shorter than max_size = 50 song, select entire playmist
    # and pad later. Otherwise, randomly select a subplaylist of 50 consecutive tracks.
    def __init__(self, data_manager, indices, max_size=50, n_pos=3, n_neg=10):
        super().__init__(data_manager, indices, max_size, n_pos, n_neg)
        self.album_covers_embs_dir = data_manager.album_covers_embs_dir
        self.track_id_to_album_uri = data_manager.track_id_to_album_uri
        self.album_covers_embs_size = data_manager.album_covers_embs_size

    def __getitem__(self, index):
        # 1-indexed for some reason, we will keep it since this is the way it is in the original code
        tracks_ids = self.data[index].indices + 1
        tracks_order = self.data[index].data
        ordered_tracks_ids = np.array([track_id for track_idx, track_id in sorted(zip(tracks_order, tracks_ids))])  # sort by position in the playlist, but keep song indices
        l = len(ordered_tracks_ids)
        if l <= self.max_size:
            X = ordered_tracks_ids
        else:
            start = np.random.randint(0, l - (self.max_size))
            X = ordered_tracks_ids[start:start + self.max_size]
        y_neg = self.sample_except_with_generator(self.n_neg, ordered_tracks_ids)

        X_albums_embs = self.get_albums_embs(X)
        y_neg_albums_embs = self.get_albums_embs(y_neg)

        return torch.LongTensor(X), X_albums_embs, torch.LongTensor(y_neg), y_neg_albums_embs

    def get_albums_embs(self, track_ids):
        embs = []
        for track_id in track_ids:
            try:
                # track_id is 1-indexed here, but 0-indexed in the file names so remove 1
                emb = np.load(self.album_covers_embs_dir / f"{self.track_id_to_album_uri[track_id - 1]}.npy")
                embs.append(emb)
            except FileNotFoundError:
                embs.append(np.zeros(self.album_covers_embs_size))
        return torch.FloatTensor(np.stack(embs))