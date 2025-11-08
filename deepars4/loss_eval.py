import torch
from tqdm.auto import tqdm

def evaluate_model(model, criterion, loader, device) -> dict[str, float]:
    eval_loss = 0
    eval_mae = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(loader))
        for batch_idx, data in pbar:
            obs, targets = data["context"], data["target"]
            obs, targets = obs.to(device), targets.to(device)
            preds = model(obs)
            mu, _ = preds
            loss = criterion(preds, targets)
            mae = torch.mean(torch.abs(mu - targets))

            eval_loss += loss.item()
            eval_mae += mae.item()

            pbar.set_description(
                'Val Batch Idx: (%d/%d) | Val loss: %.3f | MAE: %.3f' %
                    (batch_idx, len(loader), eval_loss/(batch_idx+1), eval_mae/(batch_idx+1))
            )

        avg_eval_loss = eval_loss / len(loader)
        avg_eval_mae = eval_mae / len(loader)

    return {
        "loss": avg_eval_loss,
        "mae": avg_eval_mae,
    }

def evaluate_model_rmse(model, criterion, loader, device) -> dict[str, float]:
    eval_loss = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(loader))
        for batch_idx, data in pbar:
            obs, targets = data["context"], data["target"]
            obs, targets = obs.to(device), targets.to(device)
            preds = model(obs)
            loss = criterion(preds, targets)

            eval_loss += loss.item()

            pbar.set_description(
                'Val Batch Idx: (%d/%d) | Val loss: %.3f' %
                    (batch_idx, len(loader), eval_loss/(batch_idx+1))
            )

        avg_eval_loss = eval_loss / len(loader)

    return {
        "loss": avg_eval_loss,
    }

