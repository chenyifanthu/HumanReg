import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import requests, json
import wandb
import torch
import importlib

from tqdm import tqdm
from pathlib import Path
from tools import exp
from omegaconf import OmegaConf
from dataset import get_dataloader

def train_epoch():
    global epoch_idx

    net_model.train()
    net_model.hparams.is_training = True

    pbar = tqdm(train_loader, desc=f'[Ep:{epoch_idx}] Training', ncols=120)
    for batch_idx, data in enumerate(pbar):
        data = exp.to_target_device(data, net_model.device)
        optimizer.zero_grad()
        loss, _ = net_model.step(data, 'train')
        loss.backward()
        net_model.on_after_backward()
        optimizer.step()
        scheduler.step()
        pbar.set_postfix_str("Loss=%.4f" % loss.item())

def validate_epoch():
    global metric_val_best, epoch_idx

    net_model.eval()
    net_model.hparams.is_training = False

    pbar = tqdm(val_loader, desc=f'[Ep:{epoch_idx}] Validation', ncols=120)
    for batch_idx, data in enumerate(pbar):
        data = exp.to_target_device(data, net_model.device)
        with torch.no_grad():
            loss, _ = net_model.step(data, 'valid')
            pbar.set_postfix_str(f"Loss = {loss.item():.4f}")

    metrics = net_model.log_cache.get_mean_loss_dict()
    net_model.log_cache.print_format_loss()
    net_model.log_cache.clear()
    
    wandb.log({
        "learning_rate": scheduler.get_last_lr()[0],
        **metrics
    })
    
    model_state = {
        'state_dict': net_model.state_dict(),
        'epoch': epoch_idx, 'metrics': metrics
    }

    if metrics["valid/total_loss"] < metric_val_best:
        print("* Best Loss: %.4f" % metrics["valid/total_loss"])
        metric_val_best = metrics["valid/total_loss"]
        torch.save(model_state, Path(wandb.run.dir) / "best.pth")
    torch.save(model_state, Path(wandb.run.dir) / "latest.pth")
    

if __name__ == '__main__':
    NET_CFG_PATH = 'configs/desc_net_self.yaml'
    DATA_CFG_PATH = 'configs/cape128.yaml'
    CKPT_PATH = 'weights/pretrain-model.pth'
    EPOCHS = 200
    
    model_args = OmegaConf.load(NET_CFG_PATH)
    data_args = OmegaConf.load(DATA_CFG_PATH)
    model_args.batch_size = data_args.batch_size
    wandb.init(project='HumanReg', name='cape128', config=model_args,
               notes='Self-supervised fine-tuning on cape128 dataset.')
    
    net_module = importlib.import_module("models." + model_args.model).Model
    net_model = net_module(model_args)
    if CKPT_PATH:
        ckpt_data = torch.load(CKPT_PATH)
        net_model.load_state_dict(ckpt_data['state_dict'])
        print(f"Pretrained model loaded from {CKPT_PATH}.")
    
    # Load dataset
    train_loader = get_dataloader(data_args, 'train')
    val_loader = get_dataloader(data_args, 'valid')
    
    # Load training specs
    optimizers, schedulers = net_model.configure_optimizers()
    assert len(optimizers) == 1 and len(schedulers) == 1
    optimizer, scheduler = optimizers[0], schedulers[0]
    assert scheduler['interval'] == 'step'
    scheduler = scheduler['scheduler']
    
    # Move to target device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net_model = exp.to_target_device(net_model, device)
    net_model.device = device
    
    # Train and validate within a protected loop.
    metric_val_best = 1e6
    # for epoch_idx in range(EPOCHS):
    #     train_epoch()
    #     validate_epoch()
    try:
        for epoch_idx in range(EPOCHS):
            train_epoch()
            validate_epoch()
        # send_message("Training Finished", "Best Flow Loss: %.4f" % metric_val_best)
    except Exception as ex:
        print(ex)
        # send_message("Training Error", str(ex))

    wandb.finish()
