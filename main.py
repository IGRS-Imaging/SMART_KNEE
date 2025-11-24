import argparse
import os
import torch
import config
from data.dataset import get_dataloaders
from engine.trainer import train_engine
from engine.evaluator import evaluate_model
from models import LandmarkCompletionModel

def parse_args():
    parser = argparse.ArgumentParser(description="Femur Landmark Completion")
    
    # Modes
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode: train or test')
    
    # Paths (Override config if provided)
    parser.add_argument('--csv_path', type=str, default=config.LANDMARKS_CSV, help='Path to landmarks CSV')
    parser.add_argument('--edges_path', type=str, default=config.EDGES_CSV, help='Path to edges CSV')
    parser.add_argument('--checkpoint_path', type=str, default=config.CHECKPOINT_PATH, help='Path to save/load model')
    
    # Training Params
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Get Data Loaders (Train, Val, Test)
    print(f"Loading data from {args.csv_path}...")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.csv_path, 
        args.edges_path, 
        args.batch_size
    )
    print(f"Data Split -> Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Execution Mode
    if args.mode == 'train':
        # Update config defaults with args if needed
        config.NUM_EPOCHS = args.epochs
        
        train_engine(train_loader, val_loader, args)
        
    elif args.mode == 'test':
        print("\nStarting Testing...")
        if not os.path.exists(args.checkpoint_path):
            print(f"Error: No checkpoint found at {args.checkpoint_path}")
            return

        model = LandmarkCompletionModel().to(device)
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        print("Model loaded.")
        
        # Run Detailed Evaluation on Test Set
        evaluate_model(model, test_loader, device, return_detailed=True)

if __name__ == "__main__":
    main()