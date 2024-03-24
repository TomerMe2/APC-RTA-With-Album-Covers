import argparse

import numpy as np

from src.data_manager.data_manager import DataManager
from src.plot_results import confidence_interval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recos_paths", type=str, required=False,
                        help="comma separated paths to recos")
    parser.add_argument('--data_manager_path', type=str, required=True,
                        help='Path to the data manager data, the folder should "embeddings" and "data_split" folders.')

    args = parser.parse_args()
    recos_paths = args.recos_paths.split(",")

    recos = [np.load(path) for path in recos_paths]
    data_manager = DataManager(foldername=args.data_manager_path)
    test_evaluator, test_dataloader = data_manager.get_test_data("test")

    for path, rec in zip(recos_paths, recos):
        print('Computing metrics for path:', path)

        precision = test_evaluator.compute_all_precisions(rec)
        precision_ci = confidence_interval(precision)
        print(f'Precision: {round(np.mean(precision) * 100, 2)}+-{round(precision_ci * 100, 2)} %')

        recall = test_evaluator.compute_all_recalls(rec)
        recall_ci = confidence_interval(recall)
        print(f'Recall: {round(np.mean(recall) * 100, 2)}+-{round(recall_ci * 100, 2)} %')

        r_precision = test_evaluator.compute_all_R_precisions(rec)
        r_precision_ci = confidence_interval(r_precision)
        print(f'R-Precision: {round(np.mean(r_precision) * 100, 2)}+-{round(r_precision_ci * 100, 2)} %')

        ndcg = test_evaluator.compute_all_ndcgs(rec)
        ndcg_ci = confidence_interval(ndcg)
        print(f'NDCG: {round(np.mean(ndcg) * 100, 2)}+-{round(ndcg_ci * 100, 2)} %')

        clicks = test_evaluator.compute_all_clicks(rec)
        clicks_ci = confidence_interval(clicks)
        print(f'Clicks: {round(np.mean(clicks), 2)}+-{round(clicks_ci, 2)}')

        popularity = test_evaluator.compute_norm_pop(rec)
        popularity_ci = confidence_interval(popularity)
        print(f'Popularity: {round(np.mean(popularity) * 100, 2)}+-{round(popularity_ci * 100, 2)} %')

        coverage = test_evaluator.compute_cov(rec)
        print(f'Coverage: {round(coverage * 100, 2)}%')