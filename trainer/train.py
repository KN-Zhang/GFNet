from tqdm import tqdm
import torch
import torch.distributed as dist
import wandb
import configs

def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    return batch

def log_param_statistics(named_parameters, norm_type = 2):
    named_parameters = list(named_parameters)
    grads = [p.grad for n, p in named_parameters if p.grad is not None]
    weight_norms = [p.norm(p=norm_type) for n, p in named_parameters if p.grad is not None]
    names = [n for n,p in named_parameters if p.grad is not None]
    param_norm = torch.stack(weight_norms).norm(p=norm_type)
    device = grads[0].device
    grad_norms = torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads])
    nans_or_infs = torch.isinf(grad_norms) | torch.isnan(grad_norms)
    nan_inf_names = [name for name, naninf in zip(names, nans_or_infs) if naninf]
    total_grad_norm = torch.norm(grad_norms, norm_type)
    if torch.any(nans_or_infs):
        print(f"These params have nan or inf grads: {nan_inf_names}")
    wandb.log({"grad_norm": total_grad_norm.item()}, step = configs.cfg.GLOBAL_STEP)
    wandb.log({"param_norm": param_norm.item()}, step = configs.cfg.GLOBAL_STEP)

def train_step(train_batch, model, objective, optimizer, grad_scaler, grad_clip_norm = 1.,**kwargs):
    optimizer.zero_grad()
    out = model(train_batch)
    l = objective(out, train_batch)
    grad_scaler.scale(l).backward()
    grad_scaler.unscale_(optimizer)
    log_param_statistics(model.named_parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm) # what should max norm be?
    grad_scaler.step(optimizer)
    grad_scaler.update()
    wandb.log({"grad_scale": grad_scaler._scale.item()}, step = configs.cfg.GLOBAL_STEP)
    if grad_scaler._scale < 1.:
        grad_scaler._scale = torch.tensor(1.).to(grad_scaler._scale)
    configs.cfg.GLOBAL_STEP = configs.cfg.GLOBAL_STEP + configs.cfg.STEP_SIZE # increment global step
    return {"train_out": out, "train_loss": l.item()}

def train_k_steps_cosine(
    n_0, k, dataloader, model, objective, optimizer, lr_scheduler, grad_scaler, progress_bar=True, grad_clip_norm = 1., warmup = None, ema_model = None, pbar_n_seconds = 1,
):
    for n in tqdm(range(n_0, n_0 + k), disable=(not progress_bar) or configs.cfg.RANK > 0, mininterval=pbar_n_seconds):
        batch = next(dataloader)
        model.train(True)
        batch = to_cuda(batch)
        train_step(
            train_batch=batch,
            model=model,
            objective=objective,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            grad_scaler=grad_scaler,
            n=n,
            grad_clip_norm = grad_clip_norm,
        )

    lr_scheduler.step()
    [wandb.log({f"lr_group_{grp}": lr}) for grp, lr in enumerate(lr_scheduler.get_last_lr())]


def check_stride_mismatch_and_noncontiguous_grads_ddp(model):

    import torch.distributed as dist
    rank_zero = not dist.is_initialized() or dist.get_rank() == 0
    if not rank_zero:
        return

    print("\n[🚨 Gradient Debug: stride mismatch / non-contiguous check]")
    for name, param in model.module.named_parameters():
        if param.grad is not None:
            grad = param.grad
            param_stride = param.stride()
            grad_stride = grad.stride()
            if param_stride != grad_stride:
                print(f"[⚠️ STRIDE MISMATCH] {name}: param {param_stride} vs grad {grad_stride}")
            if not grad.is_contiguous():
                print(f"[⚠️ NON-CONTIGUOUS GRAD] {name}: shape={grad.shape}, stride={grad_stride}")
