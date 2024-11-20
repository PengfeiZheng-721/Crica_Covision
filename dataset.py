# dataset.py

import os
import random
from collections import defaultdict
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

class CovisionDataset(Dataset):
    """
    Dataset class for loading images and labels for covision detection.
    Implements sequential sampling across subscenes, sampling images per batch.
    """
    def __init__(self, dataset_list: List[Tuple[str, str]], transform=None):
        """
        Args:
            dataset_list (List[Tuple[str, str]]): List of tuples where each tuple contains
                                                  (csv_file, root_dir) for a dataset.
            transform: Optional transform to be applied on an image.
        """
        self.transform = transform

        # Data structures to hold the image IDs and labels
        self.scenes = defaultdict(lambda: defaultdict(list))  # scenes[scene][subscene] = [image IDs]
        self.labels = {}  # labels[(img1_id, img2_id)] = label
        self.image_to_scene = {}  # Mapping from image ID to scene name
        self.image_id_map = {}  # Mapping from image path to unique ID
        self.id_to_image = {}   # Mapping from unique ID to image path
        self.next_image_id = 0

        for dataset_idx, (csv_file, root_dir) in enumerate(dataset_list):
            # Read CSV file and populate data structures
            with open(csv_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) != 3:
                        print(f"Skipping invalid line: {line.strip()}")
                        continue  # Skip invalid lines
                    img1_rel_path, img2_rel_path, label = parts[0], parts[1], int(parts[2])

                    # Resolve the full paths
                    img1_full_path = os.path.abspath(os.path.normpath(os.path.join(root_dir, img1_rel_path)))
                    img2_full_path = os.path.abspath(os.path.normpath(os.path.join(root_dir, img2_rel_path)))

                    # Check if images exist
                    if not os.path.exists(img1_full_path) or not os.path.exists(img2_full_path):
                        print(f"Image not found: {img1_full_path} or {img2_full_path}")
                        continue

                    # Assign unique IDs to images
                    img1_id = self.get_image_id(img1_full_path)
                    img2_id = self.get_image_id(img2_full_path)

                    # Extract scene and subscene from image paths
                    # Expected path format: .../More_vis/Scene/Subscene/saved_obs/image.png
                    img1_parts = img1_rel_path.strip('./').split('/')
                    img2_parts = img2_rel_path.strip('./').split('/')

                    # Ensure the paths have the expected structure
                    if len(img1_parts) < 6 or len(img2_parts) < 6:
                        print(f"Skipping invalid path: {img1_rel_path} or {img2_rel_path}")
                        continue  # Skip if the path doesn't match expected format

                    # Extract scene and subscene
                    img1_scene = img1_parts[2]
                    img1_subscene = img1_parts[3]
                    img2_scene = img2_parts[2]
                    img2_subscene = img2_parts[3]

                    # Ensure both images are from the same subscene
                    if img1_scene != img2_scene or img1_subscene != img2_subscene:
                        print(f"Images from different subscenes: {img1_rel_path}, {img2_rel_path}")
                        continue  # Skip pairs from different subscenes

                    # Prepend dataset ID to scene name to ensure uniqueness
                    scene = f"dataset{dataset_idx}_{img1_scene}"
                    subscene = img1_subscene

                    # Add image IDs to scenes dictionary
                    if img1_id not in self.scenes[scene][subscene]:
                        self.scenes[scene][subscene].append(img1_id)
                        self.image_to_scene[img1_id] = scene
                    if img2_id not in self.scenes[scene][subscene]:
                        self.scenes[scene][subscene].append(img2_id)
                        self.image_to_scene[img2_id] = scene

                    # Store the label
                    self.labels[(img1_id, img2_id)] = label
                    self.labels[(img2_id, img1_id)] = label  # Ensure symmetry

        # Create a list of all subscenes for sequential sampling
        self.all_subscenes = []
        for scene in self.scenes:
            for subscene in self.scenes[scene]:
                self.all_subscenes.append((scene, subscene))

        # Initialize current subscene index for sequential sampling
        self.current_subscene_idx = 0

        # Shuffle subscenes initially for randomness
        random.shuffle(self.all_subscenes)

        # Create a list of all images (image IDs) for __len__ method
        self.all_images = list(self.image_id_map.values())

        print(f"Total scenes loaded: {len(self.scenes)}")
        print(f"Total subscenes loaded: {len(self.all_subscenes)}")
        if not self.scenes:
            print("Warning: No scenes loaded. Please check the CSV files and paths.")

    def get_image_id(self, img_full_path):
        """
        Assign a unique ID to each image path.

        Args:
            img_full_path (str): Full path to the image.

        Returns:
            img_id (int): Unique image ID.
        """
        if img_full_path not in self.image_id_map:
            self.image_id_map[img_full_path] = self.next_image_id
            self.id_to_image[self.next_image_id] = img_full_path
            self.next_image_id += 1
        return self.image_id_map[img_full_path]

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        # Not used since we'll use custom sampling
        raise NotImplementedError("Use the 'sample_batch' method to get data.")

    def sample_batch(self, batch_size: int) -> Tuple[List[str], torch.Tensor]:
        """
        Samples a batch of images from subscenes to construct a batch of specified size.
        If the current subscene doesn't have enough images, images from subsequent subscenes
        are used to fill up the batch.

        Args:
            batch_size (int): Number of images per batch.

        Returns:
            Tuple[List[str], torch.Tensor]: Image paths and label matrix.
        """
        if not self.all_subscenes:
            raise ValueError("No subscenes available for sampling.")

        sampled_image_ids = []
        image_subscenes = []  # Keep track of which subscene each image comes from

        # Keep sampling images until we have enough
        while len(sampled_image_ids) < batch_size:
            # Get the current subscene
            scene, subscene = self.all_subscenes[self.current_subscene_idx]
            images = self.scenes[scene][subscene]  # List of image IDs

            # Determine number of images to sample from this subscene
            remaining_slots = batch_size - len(sampled_image_ids)
            num_images_to_sample = min(len(images), remaining_slots)

            # Sample images without replacement
            sampled_ids = random.sample(images, num_images_to_sample)

            sampled_image_ids.extend(sampled_ids)
            image_subscenes.extend([subscene] * num_images_to_sample)

            # Move to the next subscene
            self.current_subscene_idx = (self.current_subscene_idx + 1) % len(self.all_subscenes)

        # Construct the label matrix
        N = len(sampled_image_ids)
        label_matrix = torch.zeros((N, N), dtype=torch.float32)

        for i in range(N):
            for j in range(N):
                img1_id = sampled_image_ids[i]
                img2_id = sampled_image_ids[j]
                subscene1 = image_subscenes[i]
                subscene2 = image_subscenes[j]
                if subscene1 == subscene2:
                    # Images are from the same subscene
                    if img1_id == img2_id:
                        label_matrix[i, j] = 1.0  # Same image
                    else:
                        label = self.labels.get((img1_id, img2_id), 0)
                        label_matrix[i, j] = label
                else:
                    # Images are from different subscenes
                    label_matrix[i, j] = 0.0  # No label information between different subscenes

        # Convert image IDs back to paths
        image_paths = [self.id_to_image[img_id] for img_id in sampled_image_ids]

        return image_paths, label_matrix

    def load_images(self, image_paths: List[str]) -> torch.Tensor:
        """
        Loads images from the given paths and applies transformations.

        Args:
            image_paths (List[str]): List of image paths.

        Returns:
            images (torch.Tensor): Tensor of images.
        """
        images = []
        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Create a dummy image (e.g., black image) in case of failure
                image = Image.new('RGB', (518, 518), (0, 0, 0))
            if self.transform:
                image = self.transform(image)
            images.append(image)

        images = torch.stack(images)  # Shape: (B, C, H, W)
        return images
