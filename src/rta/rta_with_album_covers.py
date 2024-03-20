import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_manager.data_manager import pad_collate
from src.rta.rta_model import RTAModel
from src.rta.sequential_train_dataset_with_album_covers_embeddings import \
    SequentialTrainDatasetWithAlbumCoversEmbeddings


class RTAWithAlbumCovers(RTAModel):

    def __init__(self, representer, aggregator, data_manager, training_params={}):
        super().__init__(representer, aggregator, data_manager, training_params)

    def prepare_training_objects(self, tuning=False):
        # Prepare the optimizer, the scheduler and the data_loader that will be used for training
        optimizer = torch.optim.SGD(self.parameters(), lr=self.training_params['lr'],
                                    weight_decay=self.training_params['wd'], momentum=self.training_params['mom'],
                                    nesterov=self.training_params['nesterov'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.training_params['patience'],
                                                    gamma=self.training_params['factor'], last_epoch=- 1, verbose=False)
        if tuning:
            train_indices = self.data_manager.train_indices
        else:
            train_indices = np.concatenate((self.data_manager.train_indices, self.data_manager.val_indices))
        train_dataset = SequentialTrainDatasetWithAlbumCoversEmbeddings(self.data_manager, train_indices,
                                               max_size=self.training_params['max_size'],
                                               n_neg=self.training_params['n_neg'])
        # TODO: REPLACE pad_collate with my collate to match 4 outputs!
        train_dataloader = DataLoader(train_dataset, batch_size=self.training_params['batch_size'], shuffle=True,
                                      collate_fn=pad_collate, num_workers=0)
        return optimizer, scheduler, train_dataloader
