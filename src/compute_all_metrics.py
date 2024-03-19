import argparse

import numpy as np

from src.data_manager.data_manager import DataManager
from src.plot_results import confidence_interval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recos_path", type=str, required=False,
                        help="path to recos", default="resources/recos")
    parser.add_argument("--plots_path", type=str, required=False,
                        help="path to plots", default="resources/plots")
    parser.add_argument("--models", type=str, required=False,
                        help="comma separated names of models to evaluate",
                        default="SKNN,VSKNN,STAN,VSTAN,MF-AVG,MF-CNN,MF-GRU,MF-Transformer,FM-Transformer,NN-Transformer")
    parser.add_argument('--data_manager_path', type=str, required=True,
                        help='Path to the data manager data, the folder should "embeddings" and "data_split" folders.')

    args = parser.parse_args()
    model_names = args.models.split(",")
    l = len(model_names)
    recos = [np.load(("%s/%s.npy") % (args.recos_path, m)) for m in model_names]
    data_manager = DataManager(foldername=args.data_manager_path)
    test_evaluator, test_dataloader = data_manager.get_test_data("test")

    for model_name, rec in zip(model_names, recos):
        print('Computing metrics for model:', model_name)

        recall = test_evaluator.compute_all_recalls(rec)
        recall_ci = confidence_interval(recall)
        print(f'Recall: {np.mean(recall)}+-{recall_ci}')

        ndcg = test_evaluator.compute_all_ndcgs(rec)
        ndcg_ci = confidence_interval(ndcg)
        print(f'NDCG: {np.mean(ndcg)}+-{ndcg_ci}')

        clicks = test_evaluator.compute_all_clicks(rec)
        clicks_ci = confidence_interval(clicks)
        print(f'Clicks: {np.mean(clicks)}+-{clicks_ci}')

        precision = test_evaluator.compute_all_precisions(rec)
        precision_ci = confidence_interval(precision)
        print(f'Precision: {np.mean(precision)}+-{precision_ci}')

        r_precision = test_evaluator.compute_all_R_precisions(rec)
        r_precision_ci = confidence_interval(r_precision)
        print(f'R-Precision: {np.mean(r_precision)}+-{r_precision_ci}')

        popularity = test_evaluator.compute_norm_pop(rec)
        popularity_ci = confidence_interval(popularity)
        print(f'Popularity: {np.mean(popularity)}+-{popularity_ci}')

        coverage = test_evaluator.compute_cov(rec)
        print(f'Coverage: {coverage * 100}%')