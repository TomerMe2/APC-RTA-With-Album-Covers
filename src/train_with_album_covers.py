import json, argparse, os, time, datetime

import numpy as np
from clearml import Task

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
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of model to train")
    parser.add_argument("--params_file", type=str, required=False,
                        help="File for hyperparameters", default="resources/params/best_params_rta.json")
    parser.add_argument("--recos_path", type=str, required=False,
                        help="Path to save recos", default="resources/recos")
    parser.add_argument("--models_path", type=str, required=False,
                        help="Path to save models", default="resources/models")
    parser.add_argument('--data_manager_path', type=str, required=True,
                        help='Path to the data manager data, the folder should "embeddings" and "data_split" folders.')
    parser.add_argument('--albums_covers_embs_algorithm', type=str, required=True,
                        help='The algorithm used to compute the album covers embeddings. It can be "dinov2" or "clip".')
    parser.add_argument('--adapter_emb_size', type=int, required=True, help='The size of the adapter embedding')
    parser.add_argument('--run_name', type=str, required=False, default="default")
    parser.add_argument('--debug', action='store_true', help='Debug mode', default=False)
    args = parser.parse_args()

    Task.init(project_name="Represent-Than-Aggregate-Album-Covers", task_name=args.run_name)
    data_manager = DataManager(foldername=args.data_manager_path,
                               albums_covers_embs_algorithm=args.albums_covers_embs_algorithm)

    with open(args.params_file, "r") as f:
        p = json.load(f)

    tr_params = p[args.model_name]
    album_cover_emb_dim = data_manager.album_covers_embs.shape[1]

    if args.model_name == "MF-Transformer":
        non_album_covers_representer = BaseEmbeddingRepresenter(data_manager, tr_params['d'])
        representer = ConcatAlbumCoverEmbRepresenter(non_album_covers_representer, args.adapter_emb_size, data_manager)
        aggregator = DecoderModel(embd_size=tr_params["d"] + representer.album_cover_emb_dim,
                                  max_len=tr_params["max_size"],
                                  n_head=tr_params["n_heads"], n_layers=tr_params["n_layers"],
                                  drop_p=tr_params["drop_p"])

    elif args.model_name == "FM-Transformer":
        # TODO: remove
        raise ValueError('not supported for album covers yet')
        representer = FMRepresenter(data_manager, tr_params['d'])
        aggregator = DecoderModel(embd_size=tr_params["d"], max_len=tr_params["max_size"], n_head=tr_params["n_heads"],
                                  n_layers=tr_params["n_layers"], drop_p=tr_params["drop_p"])

    elif args.model_name == "NN-Transformer":
        # TODO: remove
        raise ValueError('not supported for album covers yet')
        representer = AttentionFMRepresenter(data_manager, emb_dim=tr_params['d'], n_att_heads=tr_params['n_att_heads'],
                                             n_att_layers=tr_params["n_att_layers"], dropout_att=tr_params["drop_att"])
        aggregator = DecoderModel(embd_size=tr_params["d"], max_len=tr_params["max_size"], n_head=tr_params["n_heads"],
                                  n_layers=tr_params["n_layers"], drop_p=tr_params["drop_p"])

    else:
        raise ValueError("Model name not recognized")

    rta_model = RTAWithAlbumCovers(data_manager, representer, aggregator, training_params=tr_params).to(get_device())
    print("Train model %s" % args.model_name)

    savePath = None if args.debug else "%s/%s_%s" % (args.models_path, args.run_name, args.model_name)

    start_fit = time.time()
    rta_model.run_training(tuning=False, savePath=savePath)
    end_fit = time.time()
    print("Model %s trained in %s " % (args.model_name, str(end_fit - start_fit)))

    test_evaluator, test_dataloader = data_manager.get_test_data("test")
    recos = rta_model.compute_recos(test_dataloader)
    end_predict = time.time()
    print("Model %s inferred in %s " % (args.model_name, str(end_predict - end_fit)))

    os.makedirs(args.recos_path, exist_ok=True)
    np.save("%s/%s_%s" % (args.recos_path, args.run_name, args.model_name), recos)
