import torch
from torch import nn
import time
from torch.utils.data import DataLoader
from typing import Dict
from pfns.bar_distribution import FullSupportBarDistribution
import schedulefree
import os

from tfmplayground.callbacks import Callback
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.utils import get_default_device


def train(model: NanoTabPFNModel, prior: DataLoader, criterion: nn.CrossEntropyLoss | FullSupportBarDistribution,
          epochs: int, accumulate_gradients: int = 1, lr: float = 1e-4, device: torch.device = None,
          callbacks: list[Callback] = None, ckpt: Dict[str, torch.Tensor] = None, multi_gpu: bool = False,
          run_name: str = 'nanoTFM', workdir: str = '.', lambda_mlm: float = 0.0):
    """
    Trains our model on the given prior using the given criterion.

    Args:
        model: (NanoTabPFNModel) our PyTorch model
        prior: (DataLoader) torch-compatible dataloader
        criterion: (nn.CrossEntropyLoss | FullSupportBarDistribution) our loss criterion
        epochs: (int) the number of epochs we train for, the number of steps that constitute an epoch are decided by the prior
        accumulate_gradients: (int) the number of gradients to accumulate before updating the weights
        device: (torch.device) the device we are using
        callbacks: A list of callback instances to execute at the end of each epoch. These can be used for
            logging, validation, or other custom actions.
        ckpt (Dict[str, torch.Tensor], optional): A checkpoint dictionary containing the model and optimizer states,
            as well as the last completed epoch. If provided, training resumes from this checkpoint.

    Returns:
        (torch.Tensor) a tensor of shape (num_rows, batch_size, num_features, embedding_size)
    """
    work_dir = os.path.join(workdir, run_name)
    os.makedirs(work_dir, exist_ok=True)
    if multi_gpu:
        model = nn.DataParallel(model)
    if callbacks is None:
        callbacks = []
    if not device:
        device = get_default_device()
    model.to(device)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    if ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    classification_task = isinstance(criterion, nn.CrossEntropyLoss)
    regression_task = not classification_task

    assert prior.num_steps % accumulate_gradients == 0, 'num_steps must be divisible by accumulate_gradients'

    try:
        for epoch in range(ckpt['epoch'] + 1 if ckpt else 1, epochs + 1):
            epoch_start_time = time.time()
            model.train()  # Turn on the train mode
            optimizer.train()
            total_loss = 0.
            total_mlm_loss = 0.
            for i, full_data in enumerate(prior):
                single_eval_pos = full_data['single_eval_pos']

                # Store original unmasked features for MLM loss computation
                x_original = full_data['x'].clone() if lambda_mlm > 0 else None

                data = (
                    full_data['x'].to(device),
                    full_data['y'][:, :single_eval_pos].to(device),
                    full_data['adj'].to(device),
                )

                # Check for NaN in targets only (features may have NaN for masking)
                if torch.isnan(data[1]).any():
                    continue

                targets = full_data['target_y'].to(device)

                if regression_task:
                    y_mean = data[1].mean(dim=1, keepdim=True)
                    y_std = data[1].std(dim=1, keepdim=True) + 1e-8
                    y_norm = (data[1] - y_mean) / y_std
                    data = (data[0], y_norm, data[2])

                # Pass original features if MLM is enabled
                if lambda_mlm > 0 and x_original is not None:
                    model_output = model(data, single_eval_pos=single_eval_pos, x_src_original=x_original.to(device))
                    if isinstance(model_output, tuple):
                        output, mlm_loss = model_output
                    else:
                        output = model_output
                        mlm_loss = torch.tensor(0.0, device=device)
                else:
                    output = model(data, single_eval_pos=single_eval_pos)
                    mlm_loss = torch.tensor(0.0, device=device)

                targets = targets[:, single_eval_pos:]
                if regression_task:
                    targets = (targets - y_mean) / y_std
                if classification_task:
                    targets = targets.reshape((-1,)).to(torch.long)
                    output = output.view(-1, output.shape[-1])

                # Compute supervised loss
                supervised_losses = criterion(output, targets)
                supervised_loss = supervised_losses.mean()

                # Combine losses
                total_batch_loss = supervised_loss + lambda_mlm * mlm_loss
                loss = total_batch_loss / accumulate_gradients

                if torch.isnan(loss):
                    print('Loss is NaN, stopping training batch.')
                    return full_data
                loss.backward()
                total_loss += supervised_loss.cpu().detach().item()
                total_mlm_loss += mlm_loss.cpu().detach().item()

                if (i + 1) % accumulate_gradients == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    optimizer.zero_grad()

            end_time = time.time()
            mean_loss: float = total_loss / len(prior)
            mean_mlm_loss: float = total_mlm_loss / len(prior)
            if lambda_mlm > 0:
                print(f'Epoch {epoch}: Supervised Loss = {mean_loss:.4f}, MLM Loss = {mean_mlm_loss:.4f}')
            model.eval()
            optimizer.eval()

            training_state = {
                'epoch': epoch,
                'architecture': {
                    'num_layers': int((model.module if multi_gpu else model).num_layers),
                    'embedding_size': int((model.module if multi_gpu else model).embedding_size),
                    'num_attention_heads': int((model.module if multi_gpu else model).num_attention_heads),
                    'mlp_hidden_size': int((model.module if multi_gpu else model).mlp_hidden_size),
                    'num_outputs': int((model.module if multi_gpu else model).num_outputs)
                },
                'model': (model.module if multi_gpu else model).state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(training_state, work_dir+'/latest_checkpoint.pth')

            for callback in callbacks:
                if type(criterion) is FullSupportBarDistribution:
                    callback.on_epoch_end(epoch, end_time - epoch_start_time, mean_loss, (model.module if multi_gpu else model), dist=criterion)
                else:
                    callback.on_epoch_end(epoch, end_time - epoch_start_time, mean_loss, (model.module if multi_gpu else model))
    except KeyboardInterrupt:
        pass
    finally:
        for callback in callbacks:
            callback.close()

    return (model.module if multi_gpu else model), total_loss
