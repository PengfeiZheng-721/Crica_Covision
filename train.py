# train.py

import os
import argparse
import torch
import torch.optim as optim
from torchvision import transforms
import random
import copy
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, roc_auc_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image
import math  # Imported math module to fix the NameError

from dataset import CovisionDataset
from network import CricaVPRNet
from loss import compute_loss

import wandb

def create_label_matrix_visualization(image_paths, label_matrix, output_path):
    """
    Create a visualization of the label matrix with images on the axes.

    Args:
        image_paths (list): List of image paths.
        label_matrix (np.ndarray): Label matrix.
        output_path (str): Path to save the visualization image.
    """
    batch_size = len(image_paths)
    fig_size = 2 * batch_size  # Adjust as needed
    fig = plt.figure(figsize=(fig_size, fig_size), constrained_layout=True)  # Enable automatic layout adjustment
    gs = gridspec.GridSpec(batch_size + 1, batch_size + 1, figure=fig, wspace=0.1, hspace=0.1)

    # Add a blank plot in the top-left corner
    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')

    # Plot images in the top row
    for i in range(batch_size):
        ax = fig.add_subplot(gs[0, i + 1])
        img = Image.open(image_paths[i])
        img = img.resize((100, 100))  # Adjust thumbnail size as needed
        ax.imshow(img)
        ax.axis('off')

    # Plot images in the left column
    for i in range(batch_size):
        ax = fig.add_subplot(gs[i + 1, 0])
        img = Image.open(image_paths[i])
        img = img.resize((100, 100))  # Adjust thumbnail size as needed
        ax.imshow(img)
        ax.axis('off')

    # Plot the label matrix
    for i in range(batch_size):
        for j in range(batch_size):
            ax = fig.add_subplot(gs[i + 1, j + 1])
            ax.imshow(np.ones((10, 10, 3)))  # Use a dummy background
            ax.axis('off')
            label = label_matrix[i, j]
            ax.text(0.5, 0.5, str(int(label)), fontsize=12, ha='center', va='center', transform=ax.transAxes)

    plt.savefig(output_path)
    plt.close()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train CricaVPRNet with label verification and visualization')
    parser.add_argument('--output_dir', type=str, default='first_batch_output',
                        help='Directory to save the first batch outputs and images')
    args = parser.parse_args()

    # Training parameters
    dataset_list = [
        (
            "/root/autodl-tmp/CoVISIONReasoningDataset_V1_EndR=0_pivotR=2_thresh=0.001_Seed=1/train.csv",
            "/root/autodl-tmp/CoVISIONReasoningDataset_V1_EndR=0_pivotR=2_thresh=0.001_Seed=1"
        ),
        # Add more datasets if needed
    ]

    batch_size = 16
    num_epochs = 20
    learning_rate = 1e-4
    validation_split = 0.2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Define maximum gradient norm for clipping
    max_grad_norm = 1.0

    # Flags to check if the first batch has been saved
    train_saved = False
    val_saved = False

    # Initialize Weights & Biases
    wandb.init(project="OrgCricaVPR",
               name=f"Batch{batch_size}")

    # Instantiate the dataset
    full_dataset = CovisionDataset(dataset_list=dataset_list,
                                   transform=transform)

    # Get all scene names
    scenes = list(full_dataset.scenes.keys())
    total_scenes = len(scenes)
    val_size = int(total_scenes * validation_split)
    train_size = total_scenes - val_size

    # Shuffle scenes to ensure randomness
    random.shuffle(scenes)

    # Split scenes into training and validation sets
    val_scenes = scenes[:val_size]
    train_scenes = scenes[val_size:]

    # Create training and validation datasets by copying the full dataset
    train_dataset = copy.copy(full_dataset)
    val_dataset = copy.copy(full_dataset)

    # Assign scenes to respective datasets
    def get_scene_from_id(img_id):
        # Get the scene name corresponding to the image ID
        return full_dataset.image_to_scene.get(img_id, None)

    # Update labels for training and validation sets
    train_dataset.labels = {
        key: value for key, value in full_dataset.labels.items()
        if get_scene_from_id(key[0]) in train_scenes
    }
    val_dataset.labels = {
        key: value for key, value in full_dataset.labels.items()
        if get_scene_from_id(key[0]) in val_scenes
    }

    # Update all images for training and validation sets
    train_dataset.all_images = [
        img_id for img_id in full_dataset.all_images
        if get_scene_from_id(img_id) in train_scenes
    ]
    val_dataset.all_images = [
        img_id for img_id in full_dataset.all_images
        if get_scene_from_id(img_id) in val_scenes
    ]

    # Update scenes in training and validation datasets
    train_dataset.scenes = {
        scene: full_dataset.scenes[scene] for scene in train_scenes
    }
    val_dataset.scenes = {
        scene: full_dataset.scenes[scene] for scene in val_scenes
    }

    # Update all subscenes for training and validation datasets
    train_dataset.all_subscenes = []
    for scene in train_dataset.scenes:
        for subscene in train_dataset.scenes[scene]:
            train_dataset.all_subscenes.append((scene, subscene))

    val_dataset.all_subscenes = []
    for scene in val_dataset.scenes:
        for subscene in val_dataset.scenes[scene]:
            val_dataset.all_subscenes.append((scene, subscene))

    # Reset current subscene indices
    train_dataset.current_subscene_idx = 0
    val_dataset.current_subscene_idx = 0

    # Checkpoint configurations
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')

    # Path to DINOv2 ViT-B/14 pretrained weights
    foundation_model_path = "/root/tf-logs/CricaVPR/dinov2_vitb14_pretrain.pth"

    # Instantiate the model and load pretrained weights
    model = CricaVPRNet(foundation_model_path=foundation_model_path)
    model = model.to(device)

    # Freeze backbone network parameters
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Define optimizer including the new feature_projection layer
    optimizer = optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': learning_rate},
        {'params': model.aggregation.parameters(), 'lr': learning_rate},
        {'params': model.feature_projection.parameters(), 'lr': learning_rate},  # Include the new linear layer
        # Add other trainable parameters if any
    ], weight_decay=1e-4)

    # Define learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Initialize training parameters
    start_epoch = 0
    best_auc = 0.0  # Initialize best AUC score

    # Log hyperparameters to wandb
    wandb.config.update({
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "validation_split": validation_split,
    })

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        num_train_iterations = 100  # Adjust as needed

        train_progress_bar = tqdm(
            range(num_train_iterations),
            desc=f"Epoch {epoch+1}/{num_epochs} - Training"
        )
        for train_iter in train_progress_bar:
            # Sample a balanced batch
            image_paths, label_matrix = train_dataset.sample_batch(batch_size)
            images = train_dataset.load_images(image_paths)
            images = images.to(device)
            label_matrix = label_matrix.to(device)

            # Save the first training batch images and labels
            if not train_saved and epoch == start_epoch and train_iter == 0:
                # Create output directory if it doesn't exist
                os.makedirs(args.output_dir, exist_ok=True)

                # Denormalize images for saving
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)
                images_denorm = images * std + mean  # Denormalize

                # Move images to CPU and convert to numpy
                images_np = images_denorm.cpu().numpy()

                # Save images in JPG format
                train_image_paths_saved = []
                for idx, img_array in enumerate(images_np):
                    # Convert from (C, H, W) to (H, W, C)
                    img_array = np.transpose(img_array, (1, 2, 0))
                    # Clip values to [0, 1]
                    img_array = np.clip(img_array, 0, 1)
                    # Convert to uint8
                    img_array = (img_array * 255).astype(np.uint8)
                    # Save image using PIL
                    img = Image.fromarray(img_array)
                    img_filename = f'train_image_{idx}.jpg'
                    img.save(os.path.join(args.output_dir, img_filename))
                    train_image_paths_saved.append(os.path.join(args.output_dir, img_filename))

                # Save image paths to a text file
                with open(os.path.join(args.output_dir, 'train_image_paths.txt'), 'w') as f:
                    f.write("First batch of training image paths:\n")
                    for idx, path in enumerate(image_paths):
                        f.write(f"{idx}: {path}\n")

                # Save label matrix to a text file
                label_matrix_np = label_matrix.cpu().numpy()
                with open(os.path.join(args.output_dir, 'train_label_matrix.txt'), 'w') as f:
                    f.write("First batch of training label matrix:\n")
                    for row in label_matrix_np:
                        row_str = ', '.join(map(str, row))
                        f.write(f"{row_str}\n")

                # Create and save visualization
                visualization_path = os.path.join(args.output_dir, 'train_label_matrix_visualization.png')
                create_label_matrix_visualization(train_image_paths_saved, label_matrix_np, visualization_path)
                print(f"Saved first batch of training images, label matrix, and visualization to {args.output_dir}")
                train_saved = True  # Mark as saved

            # Continue training
            optimizer.zero_grad()
            # Forward pass: get feature vectors
            features = model(images)  # Shape: (B, 768)
            # Compute loss
            loss = compute_loss(features, label_matrix)
            # Backward pass and optimization
            loss.backward()
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_train_loss += loss.item()
            train_progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / num_train_iterations
        print(f"Epoch [{epoch+1}/{num_epochs}] - Average Training Loss: {avg_train_loss:.4f}")

        # Log average training loss to wandb
        wandb.log({"Average Training Loss": avg_train_loss}, step=epoch + 1)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        num_val_iterations = 20  # Adjust as needed

        # Collect predictions and labels for metrics
        all_preds = []
        all_labels = []

        val_progress_bar = tqdm(
            range(num_val_iterations),
            desc=f"Epoch {epoch+1}/{num_epochs} - Validation"
        )
        for val_iter in val_progress_bar:
            # Sample a batch
            image_paths, label_matrix = val_dataset.sample_batch(batch_size)
            images = val_dataset.load_images(image_paths)
            images = images.to(device)
            label_matrix = label_matrix.to(device)

            # Save the first validation batch images and labels
            if not val_saved and epoch == start_epoch and val_iter == 0:
                # Create output directory if it doesn't exist
                os.makedirs(args.output_dir, exist_ok=True)

                # Denormalize images for saving
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)
                images_denorm = images * std + mean  # Denormalize

                # Move images to CPU and convert to numpy
                images_np = images_denorm.cpu().numpy()

                # Save images in JPG format
                val_image_paths_saved = []
                for idx, img_array in enumerate(images_np):
                    # Convert from (C, H, W) to (H, W, C)
                    img_array = np.transpose(img_array, (1, 2, 0))
                    # Clip values to [0, 1]
                    img_array = np.clip(img_array, 0, 1)
                    # Convert to uint8
                    img_array = (img_array * 255).astype(np.uint8)
                    # Save image using PIL
                    img = Image.fromarray(img_array)
                    img_filename = f'val_image_{idx}.jpg'
                    img.save(os.path.join(args.output_dir, img_filename))
                    val_image_paths_saved.append(os.path.join(args.output_dir, img_filename))

                # Save image paths to a text file
                with open(os.path.join(args.output_dir, 'val_image_paths.txt'), 'w') as f:
                    f.write("First batch of validation image paths:\n")
                    for idx, path in enumerate(image_paths):
                        f.write(f"{idx}: {path}\n")

                # Save label matrix to a text file
                label_matrix_np = label_matrix.cpu().numpy()
                with open(os.path.join(args.output_dir, 'val_label_matrix.txt'), 'w') as f:
                    f.write("First batch of validation label matrix:\n")
                    for row in label_matrix_np:
                        row_str = ', '.join(map(str, row))
                        f.write(f"{row_str}\n")

                # Create and save visualization
                visualization_path = os.path.join(args.output_dir, 'val_label_matrix_visualization.png')
                create_label_matrix_visualization(val_image_paths_saved, label_matrix_np, visualization_path)
                print(f"Saved first batch of validation images, label matrix, and visualization to {args.output_dir}")
                val_saved = True  # Mark as saved

            # Forward pass
            with torch.no_grad():
                features = model(images)  # Shape: (B, 768)
                # Compute loss
                loss = compute_loss(features, label_matrix)

            total_val_loss += loss.item()
            val_progress_bar.set_postfix(loss=loss.item())

            # Collect predictions and labels for metrics
            # Compute dot product and scaling
            N, D = features.shape
            dot_product_matrix = torch.matmul(features, features.T)  # Shape: (N, N)
            scaling_factor = 1.0 / math.sqrt(D)  # math.sqrt is now defined
            scaled_dot_product = dot_product_matrix * scaling_factor  # Shape: (N, N)

            # Apply sigmoid to scaled dot product to get probabilities
            similarity_probs = torch.sigmoid(scaled_dot_product)  # Shape: (N, N)

            # Flatten similarity probabilities and labels
            sim_scores = similarity_probs.view(-1).cpu().numpy()
            labels = label_matrix.view(-1).cpu().numpy()

            all_preds.extend(sim_scores)
            all_labels.extend(labels)

        avg_val_loss = total_val_loss / num_val_iterations

        # Compute additional metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Binarize predictions using a threshold (e.g., 0.5)
        binarized_preds = (all_preds >= 0.5).astype(int)

        # Compute accuracy
        accuracy = accuracy_score(all_labels, binarized_preds)

        # Compute precision
        precision = precision_score(all_labels, binarized_preds, zero_division=0)

        # Compute F1 score
        f1 = f1_score(all_labels, binarized_preds, zero_division=0)

        # Compute ROC AUC
        try:
            auc_score = roc_auc_score(all_labels, all_preds)
        except ValueError:
            auc_score = float('nan')  # Handle cases where AUC cannot be computed

        print(f"Epoch [{epoch+1}/{num_epochs}] - Average Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, F1 Score: {f1:.4f}, AUC: {auc_score:.4f}")

        # Log validation metrics to wandb
        wandb.log({
            "Epoch": epoch + 1,
            "Validation Loss": avg_val_loss,
            "Validation Accuracy": accuracy,
            "Validation Precision": precision,
            "Validation F1 Score": f1,
            "Validation ROC AUC": auc_score,
        }, step=epoch + 1)

        # Save the best model based on AUC
        if not np.isnan(auc_score) and auc_score > best_auc:
            best_auc = auc_score
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated with AUC: {best_auc:.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_auc': best_auc,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1}")

        # Update the learning rate scheduler
        scheduler.step()

        # Continue training if the first batch has been saved
        if train_saved and val_saved:
            print("First training and validation batches have been saved. Continuing training...")

    print(f"Training complete.")
    wandb.finish()  # End wandb run

if __name__ == '__main__':
    main()
