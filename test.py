import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from model.unet import UNet
from utils.model_utils import test_arg_parser, set_seed
from utils.data_utils import MadisonStomach
from utils.viz_utils import visualize_predictions
from utils.metric_utils import compute_dice_score

def test_model(model, test_dataloader, criterion, args, save_path, epoch):
    '''
    Tests the model on the test dataset and computes the average Dice score.
    
    Args:
    - model (torch.nn.Module): The PyTorch model to test.
    - args (argparse.Namespace): Parsed arguments for device, batch size, etc.
    - save_path (str): Directory where results (e.g., metrics plot) will be saved.
    
    Functionality:
    - Sets the model to evaluation mode and iterates over the test dataset.
    - Computes the Dice score for each batch and calculates the average.
    - Saves a plot of the Dice coefficient history.
    '''
    test_loss = 0
    test_dice_score = 0

    with torch.no_grad():
        model.eval()
        for idx, (images, masks) in enumerate(tqdm(test_dataloader, desc="Testing Model: ")):
            images, masks = images.to(args.device), masks.to(args.device)
        
            outputs = model(images)
            loss = criterion(outputs, masks)

            test_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            test_dice_score += compute_dice_score(preds, masks)

            visualize_predictions(images, masks, preds, save_path, batch_idx=idx, args=args, epoch=epoch)

        test_loss /= len(test_dataloader)
        test_dice_score /= len(test_dataloader)
    
    print(f"Test Results --> test_loss: {test_loss:.4f} | dice_score: {test_dice_score:.4f}")

    pass

if __name__ == '__main__':

    torch.cuda.empty_cache()
    
    args = test_arg_parser()
    save_path = "results"
    set_seed(42)

    #Define dataset
    dataset = MadisonStomach(data_path="madison-stomach", mode="test")
    test_dataloader = DataLoader(dataset, batch_size=args.bs)

    # Define and load your model
    model = UNet(in_channels=1, out_channels=1).to(args.device)
    model = torch.load(args.model_path, map_location=args.device)

    criterion = torch.nn.BCEWithLogitsLoss()

    test_model(model=model,
                test_dataloader=test_dataloader,
                criterion=criterion,
                args=args,
                save_path=save_path,
                epoch=-1)