"""
CT-MoE training loop and evaluation utilities.
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .model import LanguageModel


def seed_all(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def eval_ppl(model: LanguageModel, loader, device: str,
             n_batches: int = 150) -> float:
    """Cross-entropy perplexity, excluding auxiliary losses."""
    model.eval()
    model.zero_usage()
    ce_sum = n = 0
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        _, _, ce = model(x.to(device), y.to(device))
        ce_sum += ce.item()
        n += 1
    model.zero_usage()
    model.train()
    return math.exp(ce_sum / n) if n > 0 else float("inf")


def collect_diagnostics(model: LanguageModel) -> dict:
    """Routing entropy, S row entropy, and S max — averaged across layers."""
    layers = [b.moe for b in model.blocks]
    return {
        "routing_entropy": float(np.mean([l.get_routing_entropy() for l in layers])),
        "s_entropy":       float(np.mean([l.last_s_entropy for l in layers])),
        "s_max":           float(np.mean([l.last_s_max     for l in layers])),
    }


def train(cfg: Config, name: str, train_loader, val_loader,
          device: str) -> tuple:
    """
    Trains one CT-MoE variant and returns (model, logs).

    logs is a dict of lists: step, loss, val_ppl, routing_entropy,
    s_entropy, s_max, lr — one entry per log_every steps.
    """
    print(f"\n{'━'*68}")
    print(f"  {name}")
    print(f"{'━'*68}")
    seed_all(cfg.seed)

    model       = LanguageModel(cfg).to(device)
    total_steps = len(train_loader) * cfg.epochs
    s_count     = sum(p.numel() for b in model.blocks
                      if hasattr(b.moe, "S_raw") for p in [b.moe.S_raw])

    print(f"  Params total : {model.n_params() / 1e6:.2f}M")
    print(f"  S params     : {s_count}  "
          f"(lr={cfg.lr * cfg.s_lr_scale:.1e} = {cfg.s_lr_scale}x base lr)")
    print(f"  Graph routing: {cfg.use_graph_routing}  |  "
          f"Graph collab: {cfg.use_graph_collab}")

    opt   = model.make_optimizer(cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        s / max(cfg.warmup_steps, 1) if s < cfg.warmup_steps
        else max(0.05, 0.5 * (1 + math.cos(
            math.pi * (s - cfg.warmup_steps) /
            max(total_steps - cfg.warmup_steps, 1)
        )))
    ))

    use_amp = device == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    logs = {k: [] for k in
            ["step", "loss", "val_ppl", "routing_entropy",
             "s_entropy", "s_max", "lr"]}

    step = 0
    for epoch in range(cfg.epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                _, total_loss, _ = model(x, y)

            opt.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()

            if step % cfg.log_every == 0:
                diag = collect_diagnostics(model)
                vp   = eval_ppl(model, val_loader, device, cfg.eval_n_batches)

                logs["step"].append(step)
                logs["loss"].append(total_loss.item())
                logs["val_ppl"].append(vp)
                logs["routing_entropy"].append(diag["routing_entropy"])
                logs["s_entropy"].append(diag["s_entropy"])
                logs["s_max"].append(diag["s_max"])
                logs["lr"].append(opt.param_groups[0]["lr"])

                print(
                    f"  Ep{epoch+1:02d} {step:6d} | "
                    f"Loss {total_loss.item():.3f} | PPL {vp:7.1f} | "
                    f"Ent {diag['routing_entropy']:.2f} | "
                    f"S_ent {diag['s_entropy']:.3f} | "
                    f"S_max {diag['s_max']:.4f} | "
                    f"LR {logs['lr'][-1]:.2e}"
                )

            step += 1

    return model, logs
