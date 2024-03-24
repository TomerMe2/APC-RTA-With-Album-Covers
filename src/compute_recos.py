import json, argparse, os, time, datetime

import numpy as np
from clearml import Task
import torch

from src.data_manager.data_manager import DataManager
from src.rta.rta_with_album_covers import RTAWithAlbumCovers
from src.rta.utils import get_device
from src.rta.aggregator.gru import GRUNet
from src.rta.aggregator.cnn import GatedCNN
from src.rta.aggregator.decoder import DecoderModel
from src.rta.aggregator.base import AggregatorBase
from src.rta.representer.base_representer import BaseEmbeddingRepresenter
from src.rta.representer.fm_representer import FMRepresenter
from src.rta.representer.attention_representer import AttentionFMRepresenter
from src.rta.representer.concat_album_cover_emb_representer import ConcatAlbumCoverEmbRepresenter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recos_path", type=str, required=False,
                        help="Path to save recos", default="resources/recos")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to save models")
    parser.add_argument('--data_manager_path', type=str, required=True,
                        help='Path to the data manager data, the folder should "embeddings" and "data_split" folders.')
    parser.add_argument('--albums_covers_embs_algorithm', type=str, required=True,
                        help='The algorithm used to compute the album covers embeddings. It can be "dinov2" or "clip".')
    args = parser.parse_args()

    data_manager = DataManager(foldername=args.data_manager_path,
                               albums_covers_embs_algorithm=args.albums_covers_embs_algorithm)

    rta_model = torch.load(args.model_path)
    # rta_model = RTAWithAlbumCovers(data_manager, representer, aggregator, training_params=tr_params)
    # torch.load(rta_model, args.model_path)
    rta_model.to(get_device())
    print("Compute recos")

    test_evaluator, test_dataloader = data_manager.get_test_data_with_album_covers("test")
    start_time = time.time()
    recos = rta_model.compute_recos(test_dataloader)
    end_predict = time.time()
    print("Model inferred in %s " % (str(end_predict - start_time)))

    os.makedirs(args.recos_path, exist_ok=True)
    np.save("%s/recos" % (args.recos_path), recos)
