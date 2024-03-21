import argparse
from pathlib import Path
import csv
from functools import partial

import numpy as np
from tqdm.contrib.concurrent import process_map


def process_single_file(file_path, emb_size: int):

    try:
        emb = np.load(file_path)

    except Exception as e:
        print(f"Error opening file {file_path}, with exception: {e}")
        emb = np.zeros((emb_size,))
    return emb


def main(embeddings_dir: str, emb_size: int):
    path = Path(embeddings_dir)
    files = sorted(list(path.glob('*.npy')))

    print('going to write albums uris to csv')
    with open(path.parent / f'album_uris_{path.stem}.csv', 'w') as f:
        writer = csv.writer(f)
        for file in files:
            writer.writerow([file.stem])
    print('done write albums uris to csv')

    process_func = partial(process_single_file, emb_size=emb_size)
    embeddings = process_map(process_func, files, chunksize=1000)
    embeddings = np.stack(embeddings)
    np.save(path.parent / f'{path.stem}_all_embs.npy', embeddings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_dir", type=str, required=True, help="Directory with embeddings npy files")
    args = parser.parse_args()
    emb_size = 512 if Path(args.embeddings_dir).stem == "clip" else 768
    main(args.embeddings_dir, emb_size)