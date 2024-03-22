import torch

from src.rta.representer.base_representer import BaseEmbeddingRepresenter


class ConcatAlbumCoverEmbRepresenter(torch.nn.Module):

    def __init__(self, non_album_covers_representor: BaseEmbeddingRepresenter):
        super(ConcatAlbumCoverEmbRepresenter, self).__init__()
        self.non_album_covers_representor = non_album_covers_representor

    def forward(self, x, album_covers_embs):
        non_album_covers_emb = self.non_album_covers_representor(x)
        return self.compute_representation_given_sub_representations(non_album_covers_emb, album_covers_embs)

    @staticmethod
    def compute_representation_given_sub_representations(non_album_cover_emb, album_cover_emb):
        if len(non_album_cover_emb.shape) == 3:
            # we have a batch of examples
            cat_dim = 2
        elif len(non_album_cover_emb.shape) == 2:
            # we have examples
            cat_dim = 1
        else:
            raise ValueError("non_album_cover_emb should have 2 or 3 dimensions")

        return torch.cat((non_album_cover_emb, album_cover_emb), dim=cat_dim)

    def compute_all_representations(self, album_cover_embs: torch.Tensor):
        non_album_covers_emb = self.non_album_covers_representor.compute_all_representations()
        zero_emb = torch.zeros(1, album_cover_embs.shape[1], device=non_album_covers_emb.device)
        # to match the pad and mask tokens of the non album covers embs
        album_cover_embs = torch.cat((zero_emb, album_cover_embs, zero_emb), dim=0)
        return self.compute_representation_given_sub_representations(non_album_covers_emb, album_cover_embs)
