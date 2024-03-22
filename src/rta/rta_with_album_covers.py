import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_manager.data_manager import pad_collate_with_album_covers
from src.rta.rta_model import RTAModel
from src.rta.sequential_train_dataset_with_album_covers_embeddings import \
    SequentialTrainDatasetWithAlbumCoversEmbeddings
from src.rta.utils import get_device


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

        train_dataloader = DataLoader(train_dataset, batch_size=self.training_params['batch_size'], shuffle=True,
                                      collate_fn=pad_collate_with_album_covers, num_workers=5)
        return optimizer, scheduler, train_dataloader

    def run_training(self, tuning=False, savePath=False):
        # Runs the training loop of the RTAModel
        if tuning:
            test_evaluator, test_dataloader = self.data_manager.get_test_data("val")
        else:
            test_evaluator, test_dataloader = self.data_manager.get_test_data("test")
        optimizer, scheduler, train_dataloader = self.prepare_training_objects(tuning)
        batch_ct = 0
        print_every = False
        if "step_every" in self.training_params.keys():
            print_every = True
        start = time.time()
        if savePath:
            torch.save(self, savePath)
        for epoch in range(self.training_params['n_epochs']):
            print("Epoch %d/%d" % (epoch, self.training_params['n_epochs']))
            print("Elapsed time : %.0f seconds" % (time.time() - start))
            for xx_pad, xx_albums_embs_pad, yy_pad_neg, yy_neg_albums_embs, x_lens in tqdm(train_dataloader):
                self.train()
                optimizer.zero_grad()
                loss = self.compute_loss_batch(xx_pad.to(get_device()), yy_pad_neg.to(get_device()))
                loss.backward()
                if self.training_params['clip']:
                    clip_grad_norm_(self.parameters(), max_norm=self.training_params['clip'], norm_type=2)
                optimizer.step()
                if print_every:
                    if batch_ct % self.training_params["step_every"] == 0:
                        scheduler.step()
                        print(loss.item())
                        recos = self.compute_recos(test_dataloader)
                        r_prec = test_evaluator.compute_all_R_precisions(recos)
                        ndcg = test_evaluator.compute_all_ndcgs(recos)
                        click = test_evaluator.compute_all_clicks(recos)
                        print("rprec : %.3f, ndcg : %.3f, click : %.3f" % (r_prec.mean(), ndcg.mean(), click.mean()))
                batch_ct += 1
            if savePath:
                torch.save(self, savePath)
        return