from pathlib import Path
import json
from typing import List
from functools import partial

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
from PIL import Image
from io import BytesIO
from tqdm.contrib.concurrent import process_map

DATA_PATH = '/sise/home/tomerlao/datasets/spotify_million_playlist_dataset/spotify_million_playlist_dataset/data/'
# DATA_PATH = r'C:\university\rec_sys\spotify_million_dataset\data'
OUTPUT_PATH = '/sise/home/tomerlao/datasets/spotify_million_playlist_dataset/spotify_million_playlist_dataset/album_covers/'
# OUTPUT_PATH = r'C:\university\rec_sys\spotify_million_dataset\album_covers'
API_BATCH_SIZE = 20  # the maximal batch size for the Spotify API
IMG_FILE_EXTENSION = '.png'
# TODO: JSON 295 AND 296 IS PROBLEMATIC. RUN IT AGAIN.
START_FROM_FILE = 0


def get_all_files():
    return sorted(list(Path(DATA_PATH).glob('*.json')))


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


def download_and_save_album_covers(start_index: int, album_ids: List[str], sp: spotipy.Spotify):
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
                image.save(Path(OUTPUT_PATH) / (album['id'] + IMG_FILE_EXTENSION))


def get_album_covers(album_ids: List[str], json_file_idx: int):
    client_credentials_manager = SpotifyClientCredentials(client_id='',
                                                          client_secret='')
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    func = partial(download_and_save_album_covers, album_ids=album_ids, sp=sp)
    process_map(func, range(0, len(album_ids), API_BATCH_SIZE),
                chunksize=1, desc=f'Downloading Covers of File {json_file_idx}', max_workers=26)


def process_album_covers_of_file(file: Path, file_idx: int):
    album_covers_uris = get_album_covers_uris(file)
    album_covers_ids = [uri.split(':')[-1] for uri in album_covers_uris]

    # Remove duplicates
    album_covers_ids = list(set(album_covers_ids))
    downloaded_albums = set([file.stem for file in Path(OUTPUT_PATH).glob('*' + IMG_FILE_EXTENSION)])
    # Remove album covers that have already been downloaded
    album_covers_ids = [album_id for album_id in album_covers_ids if album_id not in downloaded_albums]

    get_album_covers(album_covers_ids, file_idx)


def main():
    output_path = Path(OUTPUT_PATH)
    output_path.mkdir(parents=True, exist_ok=True)

    all_files = list(get_all_files())[START_FROM_FILE:]
    for idx, file in enumerate(all_files):
        process_album_covers_of_file(file, idx + START_FROM_FILE)


if __name__ == '__main__':
    main()
