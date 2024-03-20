from pathlib import Path
import argparse
from collections import Counter

from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm


def save_embeddings(embeddings, image_paths, out_path, subdir):
    for embedding, image_path in zip(embeddings, image_paths):
        embedding_file = f"{out_path / subdir / image_path.stem}"
        np.save(embedding_file, embedding.cpu().numpy())


def create_embeddings(images_dir: str, out_path: str, clip_model, clip_processor, dino_model, dino_processor,
                      batch_size, device):
    images_paths = list(Path(images_dir).glob('*.png'))
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / 'clip').mkdir(parents=True, exist_ok=True)
    (out_path / 'dinov2').mkdir(parents=True, exist_ok=True)

    existing_file_names = Counter(list([path.stem for path in list(out_path.glob('**/*.npy'))]))
    images_paths = [path for path in images_paths if path.stem not in existing_file_names or \
                    (path.stem in existing_file_names and existing_file_names[path.stem] == 1)]

    clip_model.to(device)
    dino_model.to(device)

    for start_idx in tqdm(range(0, len(images_paths), batch_size)):
        batch_image_paths = images_paths[start_idx: start_idx + batch_size]
        batch_images = []
        for image_path in batch_image_paths:
            try:
                image = Image.open(str(image_path))
                batch_images.append(image)
            except Exception as e:
                print(f"Error opening image {image_path}, with exeption: {e}")

        # Process the image for each model
        clip_inputs = clip_processor(images=batch_images, return_tensors="pt", device=device)
        clip_inputs['pixel_values'] = clip_inputs['pixel_values'].to(device)
        dino_inputs = dino_processor(images=batch_images, return_tensors="pt", device=device)
        dino_inputs['pixel_values'] = dino_inputs['pixel_values'].to(device)

        # Generate embeddings
        with torch.no_grad():
            clip_embeddings = clip_model.get_image_features(**clip_inputs)
            dino_embeddings = dino_model(**dino_inputs).last_hidden_state.mean(dim=1)

        save_embeddings(clip_embeddings, batch_image_paths, out_path, subdir='clip')
        save_embeddings(dino_embeddings, batch_image_paths, out_path, subdir='dinov2')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images in a directory and generate embeddings.')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Path to the directory containing PNG images')
    parser.add_argument('--out_path', type=str, required=True,
                        help='Path to the directory to save the embeddings')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for the inference. Default is 64.')
    args = parser.parse_args()

    # Initialize the models and processors
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    dino_model = AutoModel.from_pretrained('facebook/dinov2-base')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    create_embeddings(args.images_dir, args.out_path, clip_model, clip_processor, dino_model, dino_processor,
                      args.batch_size, device)
