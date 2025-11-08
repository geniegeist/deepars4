'''
Train an DeepARS4 model.
'''
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, SubsetRandomSampler
import wandb
import polars as pl
import yaml

import os
import argparse

from deepars4.deepars4 import DeepARS4, NegativeBinomialNLL
from deepars4.common import DummyWandb
from deepars4.loss_eval import evaluate_model
from deepars4.split import temporal_train_val_split, spatiotemporal_subset
from deepars4.optimizer import setup_optimizer, setup_warum_up_optimizer
from deepars4.train_util import plot_forecast_vs_truth
from deepars4.dataset import TileTimeSeriesDataset


from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description='PyTorch DeepAR-S4 Training Script')

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
parser.add_argument('--config', type=str, help='Path to YAML config file')

# ----------------------------------------------------------------------
# Optimizer
# ----------------------------------------------------------------------
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for non-S4 layers')
parser.add_argument('--s4_lr', type=float, default=0.001, help='Learning rate for S4 layers')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer (Adam)')
parser.add_argument('--use_scheduler', action='store_true', help='Enable learning rate scheduler')
parser.add_argument('--warm_restart', type=int, help='Enable learning rate scheduler')
parser.add_argument('--eta_min', type=float, default=0, help='Minimum learning rate for cosine annealing scheduler')

# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
parser.add_argument('--data_path', type=str, help='Path to main dataset file')
parser.add_argument('--meta_path', type=str, help='Path to dataset metadata file')

# ----------------------------------------------------------------------
# Dataloader
# ----------------------------------------------------------------------
parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader worker threads')

# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
parser.add_argument('--n_layers', type=int, default=4, help='Number of model layers')
parser.add_argument('--d_model', type=int, default=128, help='Hidden dimension of the model')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--use_prenorm', action='store_true', help='Enable pre-layer normalization')

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--samples_per_epoch', type=int, default=10000, help='Number of samples per epoch')
parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
parser.add_argument('--val_split_date', type=str, default='2025-01-01', help='Validation split date (YYYY-MM-DD)')

# ----------------------------------------------------------------------
# Checkpointing & Logging
# ----------------------------------------------------------------------
parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/ckpt.pth', help='Path to checkpoint file')
parser.add_argument('--run', default="dummy", type=str, help='Run name for logging (e.g., wandb)')
parser.add_argument('--wandb_resume_from', type=str, help='{run_id}?_step={step}')
parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to start with')
parser.add_argument('--save_checkpoints', action='store_true', help='Enable periodic checkpoint saving')

# ----------------------------------------------------------------------
# Features & Evaluation
# ----------------------------------------------------------------------
parser.add_argument('--eval_every', type=int, default=1, help='Evaluate model every N epochs')
parser.add_argument('--sample_every', type=int, default=1, help='Sample model every N epochs')

# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
parser.add_argument('--model_tag', type=str, help='Custom tag for model output directory')

args = parser.parse_args()

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Detected device: {device}')

# Config
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

path_to_data = args.data_path if args.data_path else config["data"]["data"]
path_to_meta = args.meta_path if args.meta_path else config["data"]["meta"]

with open(path_to_meta, "r") as f:
    meta = yaml.safe_load(f)

val_split_date = args.val_split_date if args.val_split_date else config["validation"]["split_date"]
mini_val_dates = config["mini_validation"]["dates"]
mini_val_tiles = config["mini_validation"]["tiles"]
sample_dates = config["sample"]["dates"]
sample_tiles = config["sample"]["tiles"]
context_length = int((60 / config["time_bin_in_min"] * 24) * config["context_window_in_days"])
samples_per_epoch = args.samples_per_epoch if args.samples_per_epoch else config["samples_per_epoch"]
eval_every = args.eval_every if args.eval_every else config["eval_every"]
sample_every = args.sample_every if args.sample_every else config["sample_every"]

d_model = args.d_model if args.d_model else config["model"]["d_model"]
n_layers = args.n_layers if args.n_layers else config["model"]["n_layers"]
dropout = args.dropout if args.dropout else config["model"]["dropout"]
prenorm = args.use_prenorm if args.use_prenorm else config["model"]["prenorm"]

# wandb logging init
use_dummy_wandb = args.run == "dummy"
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    project="deepars4", 
    name=args.run, 
    resume_from=args.wandb_resume_from if args.wandb_resume_from else None,
    config={ 
        "data": {"meta": meta}, 
        "args": args, 
        "model": { "d_model": d_model, "n_layers": n_layers, "dropout": dropout, "prenorm": prenorm, "context_length": context_length },
        "training": {
            "lr": args.lr,
            "s4_lr": args.s4_lr,
            "weight_decay": args.weight_decay,
            "eta_min": args.eta_min,
            "use_scheduler": args.use_scheduler,
            "epochs": args.epochs,
            "samples_per_epoch": samples_per_epoch,
            "val_split_date": val_split_date,
        }
    }
)

# Data
print(f'==> Preparing data..')

df = pl.read_parquet(path_to_data)
train_df, val_df = temporal_train_val_split(df, meta, val_split_date)
mini_val_df = spatiotemporal_subset(df, meta, mini_val_dates, mini_val_tiles, context_window_in_days=config["context_window_in_days"])
sample_df = spatiotemporal_subset(df, meta, sample_dates, sample_tiles, context_window_in_days=config["context_window_in_days"])

train_dataset = TileTimeSeriesDataset(train_df, meta, context_length=context_length)
val_dataset = TileTimeSeriesDataset(val_df, meta, context_length=context_length)
mini_val_dataset = TileTimeSeriesDataset(mini_val_df, meta, context_length=context_length)
sample_dataset = TileTimeSeriesDataset(sample_df, meta, context_length=context_length)


# Dataloaders
def get_train_loader():
    print("Setting up train loader")
    # Randomly select N unique indices
    indices = torch.randperm(len(train_dataset))[:samples_per_epoch]
    train_sampler = SubsetRandomSampler(indices)

    return DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)

val_loader = DataLoader(
    val_dataset, batch_size=64, shuffle=False, num_workers=args.num_workers)
mini_val_loader = DataLoader(
    mini_val_dataset, batch_size=64, shuffle=False, num_workers=args.num_workers)
sample_loader = DataLoader(
    sample_dataset, batch_size=64, shuffle=False, num_workers=args.num_workers)

# Model
print('==> Building model..')

d_input = len(meta["features"])
model = DeepARS4(
    d_input=d_input,
    d_model=d_model,
    n_layers=n_layers,
    dropout=dropout,
    prenorm=prenorm,
    lr=args.s4_lr
)

model = model.to(device)
if device == 'cuda':
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])


criterion = NegativeBinomialNLL()
optimizer, scheduler = setup_optimizer(
    model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs, eta_min=args.eta_min, warm_restart=args.warm_restart
)

###############################################################################
# Everything after this point is standard PyTorch training!
###############################################################################

ema_beta = 0.9 # EMA decay factor
total_training_time = 0
min_eval_loss = float("inf")
best_epoch = -1

print('==> Start training..')

wandb_run.watch(model, log="all")
# we run +1 iteration to save and eval at the end
for epoch in range(args.start_epoch, args.epochs + 1):
    last_epoch = epoch == args.epochs
    samples_so_far = epoch * samples_per_epoch
    smooth_train_loss = 0 # EMA of training loss
    smooth_mae = 0 # EMA of training loss

    model.eval()
    # once in a while: evaluate model
    if last_epoch or epoch % eval_every == 0:
        val_res = evaluate_model(
            model=model,
            criterion=criterion,
            loader=mini_val_loader,
            device=device,
        )
        # logging
        if val_res["loss"] < min_eval_loss:
            min_eval_loss = val_res["loss"]
            best_epoch = epoch
        wandb_run.log({
            "epoch": epoch,
            "samples_so_far": samples_so_far,
            "val_loss": val_res["loss"],
            "val_mae": val_res["mae"],
            "min_val_loss": min_eval_loss,
            "best_epoch": best_epoch,
        })

    # once in a while: sample from model
    if sample_every > 0 and (last_epoch or (epoch % sample_every == 0)):
        plot_forecast_vs_truth(
            model=model,
            loader=sample_loader,
            device=device,
            wandb_run=wandb_run,
            epoch=epoch,
        )

    # Training
    model.train()
    train_loader = get_train_loader()
    pbar = tqdm(enumerate(train_loader))
    print("Train loader setup finished")
    print(f"Epoch {epoch} learning rate: {scheduler.get_last_lr()}")
    wandb_run.log({ "last_lr": scheduler.get_last_lr(), "epoch": epoch })
    for batch_idx, data in pbar:
        samples_so_far += args.batch_size

        obs, targets = data["context"], data["target"]
        obs, targets = obs.to(device), targets.to(device)
        optimizer.zero_grad()
        mu, alpha = model(obs)
        loss = criterion(mu, alpha, targets)
        loss.backward()
        optimizer.step()

        mae = torch.mean(torch.abs(mu - targets))

        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * loss.item() # EMA the training loss
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(batch_idx + 1))

        smooth_mae = ema_beta * smooth_mae + (1 - ema_beta) * mae.item() # EMA the training loss
        debiased_smooth_mae = smooth_mae / (1 - ema_beta**(batch_idx + 1))

        pbar.set_description(
            'Train Batch Idx: (%d/%d) | Train loss: %.6f | MAE: %.6f' %
            (batch_idx, len(train_loader), debiased_smooth_loss, debiased_smooth_mae)
        )
        
        if batch_idx % 10 == 0:
            wandb_run.log({
                "epoch": epoch,
                "batch_idx": batch_idx,
                "samples_so_far": samples_so_far,
                "train_loss": debiased_smooth_loss,
                "train_mae": debiased_smooth_mae,
                "last_lr": scheduler.get_last_lr(),
            })

    if args.save_checkpoints:
        print("Save checkpoint")

        state = {
            'model': model.state_dict(),
            'epoch': epoch,
            'samples_so_far': samples_so_far,
            "min_val_loss": min_eval_loss,
            "best_epoch": best_epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        ckpt_path = f'./checkpoint/ckpt_epoch{epoch}.pth'
        torch.save(state, ckpt_path)

        if not use_dummy_wandb:
            artifact = wandb.Artifact(f'{wandb_run.id}-artifact-epoch{epoch}', type='model')
            artifact.add_file(ckpt_path)
            wandb_run.log_artifact(artifact)

    # Logging
    if args.use_scheduler:
        scheduler.step()

wandb_run.finish()
