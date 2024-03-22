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
from src.rta.utils import get_device, padded_avg


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
                                      collate_fn=pad_collate_with_album_covers, num_workers=6)
        return optimizer, scheduler, train_dataloader

    def run_training(self, tuning=False, savePath=False):
        # Runs the training loop of the RTAModel
        if tuning:
            test_evaluator, test_dataloader = self.data_manager.get_test_data_with_album_covers("val")
        else:
            test_evaluator, test_dataloader = self.data_manager.get_test_data_with_album_covers("test")
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

                device = get_device()
                loss = self.compute_loss_batch(xx_pad.to(device), yy_pad_neg.to(device),
                                               xx_albums_embs_pad.to(device), yy_neg_albums_embs.to(device))
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

    def compute_loss_batch(self, x_pos, x_neg, pos_album_embs=None, neg_album_embs=None):
        # Computes the entirety of the loss for a batch
        pad_mask = x_pos == 0

        X_pos_rep = self.representer(x_pos, pos_album_embs)
        input_rep = X_pos_rep[:, :-1, :]  # take all elements of each sequence except the last
        Y_pos_rep = X_pos_rep[:, 1:, :]  # take all elements of each sequence except the first

        X_agg = self.aggregator.aggregate(input_rep, pad_mask[:, :-1])
        X_neg_rep = self.chose_negative_examples(X_agg, x_neg, pad_mask[:, 1:], neg_album_embs)

        pos_loss = self.compute_pos_loss_batch(X_agg, Y_pos_rep, pad_mask[:, 1:])
        neg_loss = self.compute_neg_loss_batch(X_agg, X_neg_rep, pad_mask[:, 1:])
        loss = pos_loss + neg_loss
        return loss

    def chose_negative_examples(self, X_pos_rep, x_neg, pad_mask, neg_album_embs=None):
        # Negative examples are partly made of hard negatives and easy random negatives

        if neg_album_embs is None:
            raise ValueError("neg_album_embs should be provided")

        X_neg_rep = self.representer(x_neg, neg_album_embs)
        easy_neg_rep = X_neg_rep[:, :self.training_params['n_easy'], ...]

        # draw hard negatives using nearst neighbours in the first layer song embedding space
        X_rep_avg = padded_avg(X_pos_rep, ~pad_mask)
        neg_prods = torch.diagonal(X_neg_rep.matmul(X_rep_avg.T), dim1=2, dim2=0).T
        top_neg_indices = torch.topk(neg_prods, k=self.training_params['n_hard'], dim=1)[1]
        hard_indices = torch.gather(x_neg, 1, top_neg_indices)
        hard_album_embs = torch.gather(neg_album_embs, 1,
                                       top_neg_indices.unsqueeze(-1).expand(-1, -1, neg_album_embs.size(-1)))

        hard_neg_rep = self.representer(hard_indices, hard_album_embs)
        X_neg_final = torch.cat([easy_neg_rep, hard_neg_rep], dim=1)
        return X_neg_final

    def compute_recos(self, test_dataloader, n_recos=500):
        # Compute recommendations for playlist of the validation or test sel
        dev = get_device()
        n_p = len(test_dataloader.dataset)
        with torch.no_grad():
            self.eval()
        recos = np.zeros((n_p, n_recos))
        current_batch = 0

        album_covers_embs = self.data_manager.songs_album_cover_embs.to(dev)
        all_rep = self.representer.compute_all_representations(album_covers_embs)

        for X, album_covers_of_tracks in test_dataloader:
            X = X.long().to(dev)
            bs = X.shape[0]
            seq_len = X.shape[1]
            X_rep = self.representer(X, album_covers_of_tracks.to(dev))
            X_agg = self.aggregator.aggregate_single(X_rep, torch.zeros((bs, seq_len)).to(dev))
            scores = X_agg.matmul(all_rep[1:-1].T)
            scores = scores.scatter(1, X.to(dev) - 1,
                                    value=- 10 ** 3)  # make sure songs in the seed are not recommended
            coded_recos = torch.topk(scores, k=n_recos, dim=1)[1].cpu().long()
            recos[
            current_batch * test_dataloader.batch_size: current_batch * test_dataloader.batch_size + bs] = coded_recos
            current_batch += 1
        self.train()
        return recos