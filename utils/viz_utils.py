import matplotlib.pyplot as plt
from itertools import product
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import glob
import torch
import os
import random

def visualize_predictions(images, masks, preds, save_path, batch_idx, args, epoch):
    '''
    Visualizes and saves sample predictions for a given batch of images, masks, and model outputs.

    Args:
    - images (torch.Tensor): Input images (batch of tensors).
    - masks (torch.Tensor): Ground truth segmentation masks (batch of tensors).
    - outputs (torch.Tensor): Model outputs (batch of tensors).
    - save_path (str): Directory path where the visualization will be saved.
    - epoch (int): Current epoch number (for labeling the file).
    - batch_idx (int): Index of the current batch (for labeling the file).
    
    Functionality:
    - Displays and saves the first few samples from the batch, showing the input images, ground truth masks, and predicted masks.
    - Applies a sigmoid function to the outputs and uses a threshold of 0.5 to convert them to binary masks.
    '''
    fig = plt.figure(figsize=(8,6))
    fig.suptitle(f"Batch {batch_idx+1} Predictions")
    
    num_img = min(len(images), 5)
    rnd_list = list(range(len(images)))
    random.shuffle(rnd_list)
    rnd_list = rnd_list[:num_img]

    for i, idx in enumerate(rnd_list):
        img = transforms.ToPILImage()(images[idx].cpu())
        mask = transforms.ToPILImage()(masks[idx].cpu())
        output = transforms.ToPILImage()(preds[idx].cpu())

        ax1 = fig.add_subplot(3, num_img, i + 1)
        ax1.imshow(img, cmap="gray")
        ax1.axis("off")
        ax1.set_title(f"Image: {idx+1}")

        ax2 = fig.add_subplot(3, num_img, (num_img * 1) + i + 1)
        ax2.imshow(mask, cmap="gray")
        ax2.axis("off")
        ax2.set_title(f"Mask: {idx+1}")

        ax3 = fig.add_subplot(3, num_img, (num_img * 2) + i + 1)
        ax3.imshow(output, cmap="gray")
        ax3.axis("off")
        ax3.set_title(f"Predicted: {idx+1}")

    if epoch != -1:
        os.makedirs(os.path.join(save_path), exist_ok=True)
        os.makedirs(os.path.join(save_path, args.exp_id, f"pre_test_results[{epoch+1}]"), exist_ok=True)
        fig.savefig(os.path.join(save_path, args.exp_id, f"pre_test_results[{epoch+1}]", f"fig_batch[{batch_idx+1}].jpg"))
    else:
        os.makedirs(os.path.join(save_path), exist_ok=True)
        os.makedirs(os.path.join(save_path, args.exp_id, "test_results"), exist_ok=True)
        fig.savefig(os.path.join(save_path, args.exp_id, "test_results", f"fig_batch[{batch_idx+1}].jpg"))
    plt.close(fig)

    pass

def plot_train_val_history(train_loss_history, val_loss_history, save_path, args):
    '''
    Plots and saves the training and validation loss curves.

    Args:
    - train_loss_history (list): List of training loss values over epochs.
    - val_loss_history (list): List of validation loss values over epochs.
    - save_path (str): Directory path where the plot will be saved.
    - args (argparse.Namespace): Parsed arguments containing experiment details.
    
    Functionality:
    - Plots the train and validation loss curves.
    - Saves the plot as a JPG file in the specified directory.
    '''
    os.makedirs(os.path.join(save_path), exist_ok=True)
    os.makedirs(os.path.join(save_path, args.exp_id, "train_results"), exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_title("Train & Validation Losses")
    ax.plot(range(0,len(train_loss_history)), train_loss_history, label="train")
    ax.plot(range(0,len(train_loss_history)), val_loss_history, label="val")

    fig.savefig(os.path.join(save_path, args.exp_id, "train_results", f"train_val_losses_table.jpg"))
    plt.close(fig)

    pass

def plot_metric(x, label, save_path, args, metric):
    '''
    Plots and saves a metric curve over epochs.

    Args:
    - x (list): List of metric values over epochs.
    - label (str): Label for the y-axis (name of the metric).
    - save_path (str): Directory path where the plot will be saved.
    - args (argparse.Namespace): Parsed arguments containing experiment details.
    - metric (str): Name of the metric (used for naming the saved file).
    
    Functionality:
    - Plots the given metric curve.
    - Saves the plot as a JPEG file in the specified directory.
    '''
    os.makedirs(os.path.join(save_path), exist_ok=True)
    os.makedirs(os.path.join(save_path, args.exp_id, "train_results"), exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_title(f"{metric}")
    ax.set_ylim((0,1))
    ax.plot(range(0,len(x)), x, label=label)
    ax.set_ylabel(label)

    fig.savefig(os.path.join(save_path, args.exp_id, "train_results", f"{metric}_table.jpg"))
    plt.close(fig)

    pass