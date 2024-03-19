from pathlib import Path
import json
from typing import List
from functools import partial
import argparse

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
from PIL import Image
from io import BytesIO
from tqdm.contrib.concurrent import process_map

API_BATCH_SIZE = 20  # the maximal batch size for the Spotify API
IMG_FILE_EXTENSION = '.png'


def get_all_files(data_path: str):
    return sorted(list(Path(data_path).glob('*.json')))


def get_album_covers_uris(file: Path):
    print(file)
    with open(file, 'r') as fd:
        file_text = fd.read()

    file_data = json.loads(file_text)
    album_covers_uris = []
    for playlist in file_data['playlists']:
        for track in playlist['tracks']:
            album_covers_uris.append(track['album_uri'])
    return album_covers_uris


def resize_and_pad_image(image, target_size):
    # Resize the image while preserving aspect ratio
    width, height = image.size
    aspect_ratio = width / height
    target_width, target_height = target_size
    if width > height:
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(new_height * aspect_ratio)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    # Pad the image to the target size
    padded_image = Image.new("RGB", target_size, (128, 128, 128))  # gray padding
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    padded_image.paste(resized_image, (x_offset, y_offset))

    return padded_image


def download_and_save_album_covers(start_index: int, album_ids: List[str], sp: spotipy.Spotify, output_path: Path):
    batch_ids = album_ids[start_index:start_index + API_BATCH_SIZE]
    albums = sp.albums(batch_ids)['albums']

    if albums is None or len(albums) == 0:
        print(f'Problem with retrieving albums for index {start_index}')
        return

    # Retrieve album covers
    for album in albums:
        if album is not None and album['images']:  # Check if album has images
            cover_url = album['images'][0]['url']
            response = requests.get(cover_url)
            if response.status_code == 200:
                image_data = BytesIO(response.content)
                image = Image.open(image_data)
                image = resize_and_pad_image(image, (224, 224))
                image.save(output_path / (album['id'] + IMG_FILE_EXTENSION))


def get_album_covers(album_ids: List[str], json_file_idx: int, spotify_client_id: str, spotify_client_secret: str,
                     output_path: Path):
    client_credentials_manager = SpotifyClientCredentials(client_id=spotify_client_id,
                                                          client_secret=spotify_client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    func = partial(download_and_save_album_covers, album_ids=album_ids, sp=sp, output_path=output_path)
    process_map(func, range(0, len(album_ids), API_BATCH_SIZE),
                chunksize=1, desc=f'Downloading Covers of File {json_file_idx}', max_workers=26)


def process_album_covers_of_file(file: Path, file_idx: int, spotify_client_id: str, spotify_client_secret: str,
                                 output_path: Path):
    album_covers_uris = get_album_covers_uris(file)
    album_covers_ids = [uri.split(':')[-1] for uri in album_covers_uris]

    # Remove duplicates
    album_covers_ids = list(set(album_covers_ids))
    downloaded_albums = set([file.stem for file in Path(output_path).glob('*' + IMG_FILE_EXTENSION)])
    # Remove album covers that have already been downloaded
    album_covers_ids = [album_id for album_id in album_covers_ids if album_id not in downloaded_albums]

    get_album_covers(album_covers_ids, file_idx, spotify_client_id, spotify_client_secret, output_path)


def main(spotify_client_id: str, spotify_client_secret: str, data_path: str, output_path: str, start_from_file: int):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    all_files = list(get_all_files(data_path))[start_from_file:]
    for idx, file in enumerate(all_files):
        process_album_covers_of_file(file, idx + start_from_file, spotify_client_id, spotify_client_secret, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--spotify_client_id", type=str, required=True, help="Spotify client ID")
    parser.add_argument("--spotify_client_secret", type=str, required=True, help="Spotify client secret")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the MPD unzipped jsons data")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save album covers")
    parser.add_argument("--start_from_file", type=int, required=False, default=0,
                        help="Start processing from this file index, useful when resuming from a halt by Spotify API")
    args = parser.parse_args()
    main(parser.spotify_client_id, parser.spotify_client_secret, parser.data_path, parser.output_path,
         parser.start_from_file)
