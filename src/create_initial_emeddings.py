from pathlib import Path
import argparse

from src.data_manager.data_manager import DataManager
from src.embeddings.model import MatrixFactorizationModel
from src.format_rta_input import create_side_embeddings


def create_initial_embeddings(data_manager):
    print("Creating initial song embeddings")
    mf_model = MatrixFactorizationModel(data_manager, retrain=True, emb_size=128,
                                        foldername=str(Path(data_manager.foldername) / 'embeddings'))
    return


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--out_path", type=str, required=True,
                      help="The output path to put the embeddings folder in")
    out_path = Path(args.parse_args().out_path)

    data_manager = DataManager(foldername=str(out_path))
    create_initial_embeddings(data_manager)
    create_side_embeddings(data_manager)