"""
CT-MoE: Collaborative Topology Mixture-of-Experts
===================================================
Paper: "Learning Expert Collaboration Topology in Mixture-of-Experts Language Models"
Author: Ben Maor

Architecture
------------
S is a single learned nn.Parameter(N, N) per MoE layer.
  - No input dependence — one topology per layer, shared across all tokens
  - Gradients flow directly from LM loss into S_raw
  - Symmetrized + diagonal-masked + softmax: valid row distribution
  - Interpretable: visualise as collaboration heatmap at any checkpoint

Ablation variants
-----------------
  StandardMoE          — vanilla top-k MoE, no S parameter
  CT-MoE-NoCollab      — S exists, routing bias only (no message passing)
  CT-MoE-NoRouting     — S exists, message passing only (no routing bias)
  CT-MoE-Full          — S used for both routing bias and message passing

Key implementation notes
--------------------------
  Routing bias uses COLUMN sums of S, not row sums.
    Row sums of a row-stochastic matrix = 1 always → constant offset → useless.
    Column sums vary as S learns and give the router a meaningful signal.

  S_raw uses a separate AdamW parameter group at 100x the base learning rate.
    Without this boost, the 256-scalar S_raw gradient is negligible relative
    to the 65M-parameter model and S learns no structure in 10 epochs.

  All N experts are run in parallel via batched bmm (vectorised, no Python loops).

Runtime
-------
  ~3-4 hr on RTX 4070 (10 epochs × 4 variants, 65M params)
  Set EPOCHS = 5 at the bottom for a ~1.5 hr sanity run.

Setup
-----
  pip install torch sentencepiece datasets matplotlib numpy
"""

import subprocess, sys, os, math, random, pickle
from dataclasses import dataclass

def _pip(p):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])

for pkg in ["sentencepiece", "datasets", "matplotlib"]:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Installing {pkg} …"); _pip(pkg)

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

CKPT_DIR = "ct_moe_checkpoints"
FIG_DIR  = "ct_moe_figs"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)


def seed_all(s: int):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Config
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # ── Architecture ──────────────────────────────────────────────────────────
    vocab_size  : int   = 16_000
    seq_len     : int   = 256
    d_model     : int   = 512
    n_head      : int   = 8
    n_layers    : int   = 6
    num_experts : int   = 16
    top_k       : int   = 2
    dropout     : float = 0.1

    # ── CT-MoE topology ───────────────────────────────────────────────────────
    routing_scale : float = 1.5    # multiplier on column-sum routing bias
    collab_scale  : float = 1.0    # residual amplitude of message passing
    s_temp        : float = 1.0    # softmax temperature for S (lower = sharper)

    # S_raw uses a separate parameter group at s_lr_scale × base lr.
    # Without this, the 256-scalar gradient is swamped by the 65M model.
    s_lr_scale : float = 100.0

    # ── Ablation flags ────────────────────────────────────────────────────────
    is_baseline       : bool = False   # True → StandardMoE (no S parameter)
    use_graph_routing : bool = True    # S column sums bias router logits
    use_graph_collab  : bool = True    # S governs message passing

    # ── Loss ─────────────────────────────────────────────────────────────────
    lb_weight : float = 0.01           # Switch-style load balancing coefficient

    # ── Training ─────────────────────────────────────────────────────────────
    batch_size   : int   = 32
    lr           : float = 2e-4
    weight_decay : float = 0.1
    grad_clip    : float = 1.0
    epochs       : int   = 10
    warmup_steps : int   = 600
    log_every    : int   = 200
    seed         : int   = 42

    # ── Evaluation ───────────────────────────────────────────────────────────
    eval_n_batches  : int = 150
    heatmap_batches : int = 100


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Data
# ══════════════════════════════════════════════════════════════════════════════

def load_domain_texts(max_chars: int = 25_000_000):
    """
    Streams code (CodeParrot-Clean) and scientific text (arXiv-Summarization).
    Falls back to WikiText-2 if streaming fails.
    Returns (train_text, val_code_text, val_arxiv_text).
    """
    from datasets import load_dataset

    def stream(name, cfg_name, split, field, mc, label):
        print(f"  [{label}] streaming …")
        ds = load_dataset(name, cfg_name, split=split, streaming=True,
                          trust_remote_code=True)
        texts, chars = [], 0
        for item in ds:
            t = (item.get(field) or "").strip()
            if not t: continue
            texts.append(t); chars += len(t)
            if chars >= mc: break
        print(f"    → {chars:,} chars, {len(texts):,} docs")
        return texts

    try:
        code  = stream("codeparrot/codeparrot-clean", None,
                       "train", "content", max_chars, "code")
        arxiv = stream("ccdv/arxiv-summarization", "document",
                       "train", "article", max_chars, "arxiv")
    except Exception as e:
        print(f"  Domain streams failed ({e}) — falling back to WikiText-2")
        ds   = load_dataset("wikitext", "wikitext-2-raw-v1")
        join = lambda s: "\n".join(t for t in ds[s]["text"] if t.strip())
        full = join("train"); val = join("validation")
        mid  = len(val) // 2
        return full, val[:mid], val[mid:]

    def split90(lst):
        n = max(1, len(lst) // 10)
        return lst[n:], lst[:n]

    ct, cv = split90(code)
    at, av = split90(arxiv)
    join   = lambda lst: "\n\n".join(lst)
    return join(ct) + "\n\n" + join(at), join(cv), join(av)


def train_spm(text: str, vocab_size: int, prefix: str = "spm_ct_moe"):
    """Train a SentencePiece BPE tokeniser, or load if already cached."""
    import sentencepiece as spm
    if not os.path.exists(f"{prefix}.model"):
        tmp = f"{prefix}_raw.txt"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text[:4_000_000])
        spm.SentencePieceTrainer.train(
            input=tmp, model_prefix=prefix, vocab_size=vocab_size,
            character_coverage=0.9995, model_type="bpe",
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            input_sentence_size=5_000_000, shuffle_input_sentence=True)
        os.remove(tmp)
    sp = spm.SentencePieceProcessor()
    sp.load(f"{prefix}.model")
    return sp


class TokenDataset(Dataset):
    def __init__(self, text: str, sp, seq_len: int):
        self.L = seq_len
        ids = []
        for s in range(0, len(text), 500_000):
            ids.extend(sp.encode(text[s:s+500_000], out_type=int))
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return (len(self.data) - 1) // self.L

    def __getitem__(self, i):
        s = i * self.L
        c = self.data[s:s+self.L+1]
        return c[:-1], c[1:]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Model
# ══════════════════════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv    = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj   = nn.Linear(cfg.d_model,     cfg.d_model, bias=False)
        self.p      = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        hd = C // self.n_head
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def rs(t): return t.view(B, T, self.n_head, hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            rs(q), rs(k), rs(v),
            is_causal=True, dropout_p=self.p if self.training else 0.)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MoELayer(nn.Module):
    """
    Unified MoE layer supporting all four ablation variants via Config flags.

    S is a learned nn.Parameter(N, N) optimised end-to-end from the LM loss.
    - Symmetrised:  S[i,j] == S[j,i]  (collaboration is mutual)
    - Diagonal-masked before softmax:  no self-loops
    - Row-stochastic after softmax:    each row is a probability distribution

    Routing bias: uses column sums S.sum(dim=0)
      Column sums measure how much total incoming attention each expert receives.
      They vary as S learns and provide a meaningful routing signal.
      (Row sums are always 1.0 by construction — useless as a bias.)

    Message passing: extracts the k×k subgraph for the selected experts,
      re-normalises rows, then applies a weighted-average residual update.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        N, D     = cfg.num_experts, cfg.d_model

        # Expert weights and router — shared across all variants
        self.W1     = nn.Parameter(torch.randn(N, D, D) / D**0.5)
        self.W2     = nn.Parameter(torch.randn(N, D, D) / D**0.5)
        self.router = nn.Linear(D, N, bias=False)
        self.register_buffer("usage", torch.zeros(N))

        # S_raw — only instantiated for CT-MoE variants
        self.has_graph = not cfg.is_baseline
        if self.has_graph:
            # Zero init → softmax gives uniform 1/(N-1) after diagonal mask.
            # The flat prior places no assumption on which pairs should collaborate.
            self.S_raw = nn.Parameter(torch.zeros(N, N))

        # Diagnostics logged during training
        self.last_s_entropy = 0.0
        self.last_s_max     = 0.0

    # ── Topology ──────────────────────────────────────────────────────────────

    def get_S(self) -> torch.Tensor:
        """
        Derives the normalised N×N adjacency matrix from S_raw.
        Cheap: O(N²) with N=16. Called once per forward pass.
        """
        N   = self.cfg.num_experts
        dev = self.S_raw.device

        S = (self.S_raw + self.S_raw.T) / 2                          # symmetrise
        S = S.masked_fill(torch.eye(N, dtype=torch.bool, device=dev),
                          float("-inf"))                               # mask diagonal
        S = torch.softmax(S / self.cfg.s_temp, dim=-1)               # row-stochastic

        with torch.no_grad():
            ent = -(S * (S + 1e-9).log()).sum(dim=-1).mean()
            self.last_s_entropy = ent.item()
            S_nd = S.clone().fill_diagonal_(0)
            self.last_s_max = S_nd.max().item()

        return S

    # ── Load balancing ────────────────────────────────────────────────────────

    def lb_loss(self, weights: torch.Tensor,
                topk_idx: torch.Tensor) -> torch.Tensor:
        """Switch-Transformer load balancing: N · Σᵢ fᵢ · pᵢ"""
        N = self.cfg.num_experts
        f = F.one_hot(topk_idx, N).float().sum(dim=-2).mean(dim=(0, 1))
        p = weights.mean(dim=(0, 1))
        return N * (f * p).sum()

    # ── Routing ───────────────────────────────────────────────────────────────

    def route(self, x: torch.Tensor, S) -> tuple:
        B, T, D = x.shape
        logits   = self.router(x.view(B*T, D)).view(B, T, self.cfg.num_experts)

        if self.has_graph and self.cfg.use_graph_routing and S is not None:
            col_sums = S.sum(dim=0)                                   # (N,)
            logits   = logits + self.cfg.routing_scale * col_sums

        weights            = torch.softmax(logits, dim=-1)
        topk_val, topk_idx = torch.topk(weights, self.cfg.top_k, dim=-1)

        if self.training:
            self.usage.scatter_add_(
                0, topk_idx.flatten(),
                torch.ones(topk_idx.numel(), device=topk_idx.device,
                           dtype=self.usage.dtype))

        return weights, topk_val, topk_idx

    # ── Expert execution ──────────────────────────────────────────────────────

    def expert_fwd(self, x: torch.Tensor,
                   topk_idx: torch.Tensor) -> torch.Tensor:
        """
        Runs all N experts in parallel via batched matmul,
        then gathers only the top-k outputs each token used.
        Avoids Python-level loops over experts.
        """
        B, T, D = x.shape
        K, N    = self.cfg.top_k, self.cfg.num_experts

        xf      = x.view(B * T, D)
        xf_exp  = xf.unsqueeze(0).expand(N, -1, -1).contiguous()    # (N, BT, D)
        h       = F.gelu(torch.bmm(xf_exp, self.W1))                 # (N, BT, D)
        out_all = torch.bmm(h, self.W2)                              # (N, BT, D)

        out_t     = out_all.permute(1, 0, 2)                         # (BT, N, D)
        idx       = topk_idx.view(B * T, K)
        out_stack = torch.gather(
            out_t, 1, idx.unsqueeze(-1).expand(B * T, K, D))         # (BT, K, D)

        return out_stack.view(B, T, K, D)

    # ── Message passing ───────────────────────────────────────────────────────

    def interact(self, out_stack: torch.Tensor,
                 topk_idx: torch.Tensor,
                 S: torch.Tensor) -> torch.Tensor:
        """
        One step of graph message passing over the k selected experts.

        Extracts the k×k subgraph of S for each token's selected experts,
        re-normalises rows (subgraph extraction breaks row-stochastic property),
        then applies a weighted-average residual update.
        """
        B, T, K, D = out_stack.shape

        ei    = topk_idx.unsqueeze(-1).expand(B, T, K, K)            # row indices
        ej    = topk_idx.unsqueeze(-2).expand(B, T, K, K)            # col indices
        S_sub = S[ei, ej]                                             # (B, T, K, K)
        S_sub = S_sub / (S_sub.sum(dim=-1, keepdim=True) + 1e-9)     # re-normalise

        msg = torch.matmul(S_sub, out_stack)                          # (B, T, K, D)
        return out_stack + self.cfg.collab_scale * msg

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> tuple:
        S = self.get_S() if self.has_graph else None

        weights, topk_val, topk_idx = self.route(x, S)
        out_stack = self.expert_fwd(x, topk_idx)

        if self.has_graph and self.cfg.use_graph_collab and S is not None:
            out_stack = self.interact(out_stack, topk_idx, S)

        result = (out_stack * topk_val.unsqueeze(-1)).sum(dim=2)
        aux    = self.cfg.lb_weight * self.lb_loss(weights, topk_idx)
        return result, aux

    def get_routing_entropy(self) -> float:
        p = self.usage / (self.usage.sum() + 1e-9)
        ent = -(p * (p + 1e-9).log()).sum().item()
        self.usage.zero_()
        return ent

    @torch.no_grad()
    def get_S_numpy(self) -> np.ndarray:
        if not self.has_graph:
            return np.zeros((self.cfg.num_experts, self.cfg.num_experts))
        return self.get_S().cpu().numpy()


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.moe  = MoELayer(cfg)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln1(x))
        o, aux = self.moe(self.ln2(x))
        return x + o, aux


class LanguageModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg    = cfg
        self.emb    = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos    = nn.Embedding(cfg.seq_len,    cfg.d_model)
        self.drop   = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f   = nn.LayerNorm(cfg.d_model)
        self.head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.emb.weight   # weight tying
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor, targets=None):
        B, T = idx.shape
        x    = self.drop(self.emb(idx) + self.pos(torch.arange(T, device=idx.device)))
        aux  = 0.0
        for block in self.blocks:
            x, a = block(x); aux += a
        logits = self.head(self.ln_f(x))
        ce = total = None
        if targets is not None:
            ce    = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            total = ce + aux / self.cfg.n_layers
        return logits, total, ce

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def zero_usage(self):
        for b in self.blocks: b.moe.usage.zero_()

    def get_all_S(self) -> list:
        """Returns list of (N, N) numpy arrays, one per layer."""
        return [b.moe.get_S_numpy() for b in self.blocks]

    def make_optimizer(self, cfg: Config) -> torch.optim.Optimizer:
        """
        S_raw parameters get a 100x learning rate boost via a separate group.
        This is necessary because the 256-scalar S_raw gradient is negligible
        relative to the 65M-parameter model under AdamW's adaptive moments.
        """
        s_params     = [b.moe.S_raw for b in self.blocks
                        if hasattr(b.moe, "S_raw")]
        other_params = [p for p in self.parameters()
                        if not any(p is sp for sp in s_params)]

        groups = [{"params": other_params, "lr": cfg.lr}]
        if s_params:
            groups.append({
                "params": s_params,
                "lr": cfg.lr * cfg.s_lr_scale,
                "weight_decay": 0.0,
            })
        return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)

    @torch.no_grad()
    def get_routing_freq(self, loader, device, n_batches: int = 100):
        """Per-expert routing frequency over a dataset split."""
        was_training = self.training; self.train()
        self.zero_usage()
        for i, (x, _) in enumerate(loader):
            if i >= n_batches: break
            self(x.to(device))
        freq = torch.zeros(self.cfg.num_experts)
        for b in self.blocks: freq += b.moe.usage.cpu()
        self.zero_usage()
        if not was_training: self.eval()
        return freq / (freq.sum() + 1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Training
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_ppl(model: LanguageModel, loader, device,
             n_batches: int = 150) -> float:
    """Perplexity from cross-entropy loss only (excludes aux losses)."""
    model.eval(); model.zero_usage()
    ce_sum = n = 0
    for i, (x, y) in enumerate(loader):
        if i >= n_batches: break
        _, _, ce = model(x.to(device), y.to(device))
        ce_sum += ce.item(); n += 1
    model.zero_usage(); model.train()
    return math.exp(ce_sum / n) if n > 0 else float("inf")


def collect_diagnostics(model: LanguageModel) -> dict:
    layers = [b.moe for b in model.blocks]
    return {
        "entropy":   float(np.mean([l.get_routing_entropy() for l in layers])),
        "s_entropy": float(np.mean([l.last_s_entropy for l in layers])),
        "s_max":     float(np.mean([l.last_s_max for l in layers])),
    }


def train(cfg: Config, name: str, train_loader, val_loader,
          device: str):
    print(f"\n{'━'*68}")
    print(f"  {name}")
    print(f"{'━'*68}")
    seed_all(cfg.seed)

    model       = LanguageModel(cfg).to(device)
    total_steps = len(train_loader) * cfg.epochs
    s_count     = sum(p.numel() for b in model.blocks
                      if hasattr(b.moe, "S_raw") for p in [b.moe.S_raw])

    print(f"  Params total : {model.n_params()/1e6:.2f}M")
    print(f"  S params     : {s_count}  "
          f"(lr={cfg.lr * cfg.s_lr_scale:.1e} = {cfg.s_lr_scale:.0f}x base lr)")
    print(f"  Graph routing: {cfg.use_graph_routing}  |  "
          f"Graph collab: {cfg.use_graph_collab}")
    if not cfg.is_baseline:
        N = cfg.num_experts
        print(f"  S init: uniform 1/{N-1}={1/(N-1):.4f}  |  "
              f"target s_max >> {1/(N-1):.4f}")

    opt   = model.make_optimizer(cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        s / max(cfg.warmup_steps, 1) if s < cfg.warmup_steps
        else max(0.05, 0.5 * (1 + math.cos(
            math.pi * (s - cfg.warmup_steps) /
            max(total_steps - cfg.warmup_steps, 1))))
    ))

    use_amp = (device == "cuda")
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
            scaler.step(opt); scaler.update(); sched.step()

            if step % cfg.log_every == 0:
                diag = collect_diagnostics(model)
                vp   = eval_ppl(model, val_loader, device, cfg.eval_n_batches)

                logs["step"].append(step)
                logs["loss"].append(total_loss.item())
                logs["val_ppl"].append(vp)
                logs["routing_entropy"].append(diag["entropy"])
                logs["s_entropy"].append(diag["s_entropy"])
                logs["s_max"].append(diag["s_max"])
                logs["lr"].append(opt.param_groups[0]["lr"])

                print(
                    f"  Ep{epoch+1:02d} {step:6d} | "
                    f"Loss {total_loss.item():.3f} | PPL {vp:7.1f} | "
                    f"Ent {diag['entropy']:.2f} | "
                    f"S_ent {diag['s_entropy']:.3f} | "
                    f"S_max {diag['s_max']:.4f} | "
                    f"LR {logs['lr'][-1]:.2e}"
                )
            step += 1

    return model, logs


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Ablation runner
# ══════════════════════════════════════════════════════════════════════════════

VARIANTS = [
    ("CT-MoE-Full",      dict()),
    ("StandardMoE",      dict(is_baseline=True)),
    ("CT-MoE-NoCollab",  dict(use_graph_collab=False)),
    ("CT-MoE-NoRouting", dict(use_graph_routing=False)),
]

COLORS = {
    "StandardMoE":      "#888780",
    "CT-MoE-NoCollab":  "#378ADD",
    "CT-MoE-NoRouting": "#D85A30",
    "CT-MoE-Full":      "#1D9E75",
}

DISPLAY_NAMES = {
    "CT-MoE-Full":      "CT-MoE (Full)",
    "CT-MoE-NoCollab":  "CT-MoE (No Collaboration)",
    "CT-MoE-NoRouting": "CT-MoE (No Routing)",
    "StandardMoE":      "Standard MoE",
}


def run_ablation(train_loader, val_loaders: dict, device: str,
                 epochs: int = None) -> dict:
    """
    Trains all four variants sequentially.
    Saves a checkpoint per variant immediately after completion —
    if the run is interrupted, completed variants are loaded from disk
    and skipped on restart.
    """
    results = {}

    for name, overrides in VARIANTS:
        ckpt = os.path.join(CKPT_DIR, f"{name.replace('-','_')}.pkl")

        if os.path.exists(ckpt):
            print(f"\n  [{name}] loading checkpoint …")
            with open(ckpt, "rb") as f:
                results[name] = pickle.load(f)
            continue

        cfg = Config(**overrides)
        if epochs is not None:
            cfg.epochs = epochs

        model, logs = train(cfg, name, train_loader,
                            val_loaders["mixed"], device)

        print(f"\n  [{name}] Domain evaluation …")
        domain_ppl = {d: eval_ppl(model, ldr, device)
                      for d, ldr in val_loaders.items()}
        all_S      = model.get_all_S()
        freq_code  = model.get_routing_freq(val_loaders["code"],  device)
        freq_arxiv = model.get_routing_freq(val_loaders["arxiv"], device)

        results[name] = {
            "logs":       logs,
            "cfg":        cfg,
            "n_params":   model.n_params(),
            "domain_ppl": domain_ppl,
            "all_S":      all_S,
            "freq_code":  freq_code.cpu().numpy(),
            "freq_arxiv": freq_arxiv.cpu().numpy(),
        }

        with open(ckpt, "wb") as f:
            pickle.dump(results[name], f)
        print(f"  [{name}] Saved → {ckpt}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Visualisation
# ══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "lines.linewidth": 1.8, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
})


def _save(name: str):
    plt.tight_layout()
    path = f"{FIG_DIR}/{name}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def smooth(arr, w: int = 7):
    return np.convolve(arr, np.ones(w)/w, mode="same")


def fig1_val_ppl(results: dict):
    """Two-panel PPL plot: log-scale full run + zoomed linear convergence."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for name, r in results.items():
        steps = np.array(r["logs"]["step"])
        ppl   = np.array(r["logs"]["val_ppl"])
        color = COLORS[name]
        label = DISPLAY_NAMES.get(name, name)

        mask_a = steps > 200
        axes[0].semilogy(steps[mask_a], ppl[mask_a],
                         color=color, label=label, linewidth=1.6)

        cutoff = steps.max() * 0.40
        mask_b = steps > cutoff
        axes[1].plot(steps[mask_b], ppl[mask_b],
                     color=color, label=label, linewidth=1.6)

    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Validation PPL (log scale)")
    axes[0].set_title("Full training — convergence behaviour")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Validation PPL")
    axes[1].set_title("Convergence zone — zoomed linear scale")

    all_final = []
    for r in results.values():
        ppl_arr = np.array(r["logs"]["val_ppl"])
        steps   = np.array(r["logs"]["step"])
        all_final.extend(ppl_arr[steps > steps.max() * 0.40].tolist())
    pad = (max(all_final) - min(all_final)) * 0.25
    axes[1].set_ylim(min(all_final) - pad, max(all_final) + pad)

    for name, r in results.items():
        steps   = np.array(r["logs"]["step"])
        ppl_arr = np.array(r["logs"]["val_ppl"])
        axes[1].annotate(
            f"{DISPLAY_NAMES.get(name, name)}  {ppl_arr[-1]:.1f}",
            xy=(steps[-1], ppl_arr[-1]),
            xytext=(6, 0), textcoords="offset points",
            color=COLORS[name], fontsize=7.5, va="center", ha="left",
        )
    axes[1].set_xlim(axes[1].get_xlim()[0], axes[1].get_xlim()[1] * 1.35)

    _save("fig1_val_ppl")


def fig2_train_loss(results: dict):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, r in results.items():
        ax.plot(r["logs"]["step"], smooth(np.array(r["logs"]["loss"])),
                color=COLORS[name], label=DISPLAY_NAMES.get(name, name))
    ax.set_xlabel("Training step"); ax.set_ylabel("Loss (smoothed)")
    ax.set_title("Training loss — all variants")
    ax.legend()
    _save("fig2_train_loss")


def fig3_routing_entropy(results: dict):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, r in results.items():
        ax.plot(r["logs"]["step"], r["logs"]["routing_entropy"],
                color=COLORS[name], label=DISPLAY_NAMES.get(name, name))
    ax.set_xlabel("Training step"); ax.set_ylabel("Routing entropy (nats)")
    ax.set_title("Expert specialisation — routing entropy (lower = more focused)")
    ax.legend()
    _save("fig3_routing_entropy")


def fig4_S_heatmaps(results: dict):
    """Learned S per layer for each CT-MoE variant."""
    for name in ["CT-MoE-Full", "CT-MoE-NoCollab", "CT-MoE-NoRouting"]:
        if name not in results: continue
        all_S    = results[name]["all_S"]
        n_layers = len(all_S)
        if n_layers == 0: continue

        show_idx = sorted(set([0, n_layers // 2, n_layers - 1]))
        fig, axes = plt.subplots(1, len(show_idx),
                                 figsize=(5 * len(show_idx), 4.5))
        if len(show_idx) == 1: axes = [axes]

        N           = all_S[0].shape[0]
        uniform_val = 1.0 / (N - 1)

        for ax, li in zip(axes, show_idx):
            S    = all_S[li]
            vmax = max(float(S.max()), uniform_val * 2.0)
            im   = ax.imshow(S, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
            ax.set_title(f"Layer {li}")
            ax.set_xlabel("Expert j (receives)")
            ax.set_ylabel("Expert i (sends)")
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.contour(S, levels=[uniform_val], colors=["#378ADD"],
                       linewidths=0.9, linestyles="--")

        fig.suptitle(
            f"{DISPLAY_NAMES.get(name, name)} — learned S per layer\n"
            f"blue dashed = uniform baseline (1/{N-1}={uniform_val:.3f})  |  "
            f"warm regions = learned collaboration structure",
            y=1.03)
        _save(f"fig4_S_{name.replace('-','_')}")


def fig5_S_entropy(results: dict):
    fig, ax = plt.subplots(figsize=(8, 4))
    N = Config().num_experts
    uniform_ent = math.log(N - 1)
    ax.axhline(uniform_ent, color="gray", lw=0.9, ls="--",
               label=f"Uniform  log({N-1}) = {uniform_ent:.3f}")

    for name, r in results.items():
        if name == "StandardMoE": continue
        logs = r["logs"]
        if not logs.get("s_entropy"): continue
        ax.plot(logs["step"], logs["s_entropy"],
                color=COLORS[name], label=DISPLAY_NAMES.get(name, name))

    ax.set_xlabel("Training step")
    ax.set_ylabel("S row entropy (nats)")
    ax.set_title("S structure over training\n"
                 "decreasing below dashed = non-uniform topology learned")
    ax.legend()
    _save("fig5_S_entropy")


def fig6_S_max(results: dict):
    fig, ax = plt.subplots(figsize=(8, 4))
    N = Config().num_experts
    uniform_max = 1.0 / (N - 1)
    ax.axhline(uniform_max, color="gray", lw=0.9, ls="--",
               label=f"Uniform  1/{N-1} = {uniform_max:.4f}")

    for name, r in results.items():
        if name == "StandardMoE": continue
        logs = r["logs"]
        if not logs.get("s_max"): continue
        ax.plot(logs["step"], logs["s_max"],
                color=COLORS[name], label=DISPLAY_NAMES.get(name, name))

    ax.set_xlabel("Training step")
    ax.set_ylabel("Max S[i,j] (off-diagonal, mean over layers)")
    ax.set_title("Peak collaboration weight over training\n"
                 "rising above dashed = S learned strong expert pairs")
    ax.legend()
    _save("fig6_S_max")


def fig7_domain_ppl(results: dict):
    domains = ["code", "arxiv", "mixed"]
    names   = list(results.keys())
    x, w    = np.arange(len(domains)), 0.18

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(names):
        ppls = [results[name]["domain_ppl"].get(d, 0) for d in domains]
        bars = ax.bar(x + (i - 1.5) * w, ppls, w,
                      label=DISPLAY_NAMES.get(name, name),
                      color=COLORS[name], alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        for bar, ppl in zip(bars, ppls):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    f"{ppl:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x); ax.set_xticklabels(domains)
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title("Domain-specific perplexity — ablation")
    ax.legend(); ax.grid(axis="y", alpha=0.3); ax.grid(axis="x", alpha=0)
    _save("fig7_domain_ppl")


def fig8_specialization(results: dict):
    if "CT-MoE-Full" not in results: return
    r    = results["CT-MoE-Full"]
    fc   = r["freq_code"]
    fa   = r["freq_arxiv"]
    spec = (fc - fa) / (fc + fa + 1e-9)
    N    = len(spec)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    bar_colors = ["#1D9E75" if s > 0 else "#D85A30" for s in spec]
    axes[0].barh(range(N), spec, color=bar_colors, alpha=0.85,
                 edgecolor="white", linewidth=0.5)
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set_yticks(range(N))
    axes[0].set_yticklabels([f"E{i}" for i in range(N)])
    axes[0].set_xlabel("Specialisation  (−1 = arxiv, +1 = code)")
    axes[0].set_title("Expert domain specialisation — CT-MoE Full")
    axes[0].legend(handles=[
        Patch(color="#1D9E75", label="Code-leaning"),
        Patch(color="#D85A30", label="Arxiv-leaning"),
    ], loc="lower right")

    xs = np.arange(N); bw = 0.35
    axes[1].bar(xs - bw/2, fc, bw, color="#1D9E75", alpha=0.85,
                label="Code", edgecolor="white")
    axes[1].bar(xs + bw/2, fa, bw, color="#D85A30", alpha=0.85,
                label="Arxiv", edgecolor="white")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([f"E{i}" for i in range(N)], rotation=45)
    axes[1].set_ylabel("Routing frequency")
    axes[1].set_title("Per-expert routing frequency by domain")
    axes[1].legend()
    _save("fig8_specialization")


def fig9_eigenspectrum(results: dict):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    N = Config().num_experts
    ax.axhline(1.0 / N, color="gray", lw=0.8, ls="--",
               label=f"Uniform  1/{N} = {1/N:.4f}")

    for name, r in results.items():
        if name == "StandardMoE" or not r["all_S"]: continue
        S    = r["all_S"][-1]
        eigv = np.sort(np.linalg.eigvalsh(S))[::-1]
        eigv = np.abs(eigv) / (np.abs(eigv).sum() + 1e-9)
        ax.plot(range(len(eigv)), eigv, color=COLORS[name],
                marker="o", markersize=3,
                label=DISPLAY_NAMES.get(name, name))

    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("Normalised |eigenvalue|")
    ax.set_title("S eigenspectrum — final layer\n"
                 "dominant first eigenvalue = structured (low-rank) topology")
    ax.legend()
    _save("fig9_eigenspectrum")


def fig10_S_comparison(results: dict):
    """Side-by-side final-layer S for all CT-MoE variants."""
    names = [n for n in ["CT-MoE-NoCollab", "CT-MoE-NoRouting", "CT-MoE-Full"]
             if n in results and results[n]["all_S"]]
    if not names: return

    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5))
    if len(names) == 1: axes = [axes]

    N = Config().num_experts
    uniform_val = 1.0 / (N - 1)

    all_S_final = [results[n]["all_S"][-1] for n in names]
    vmax        = max(max(S.max() for S in all_S_final), uniform_val * 2.0)

    for ax, name, S in zip(axes, names, all_S_final):
        im = ax.imshow(S, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(f"{DISPLAY_NAMES.get(name, name)}\nfinal layer S")
        ax.set_xlabel("Expert j"); ax.set_ylabel("Expert i")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.contour(S, levels=[uniform_val], colors=["#378ADD"],
                   linewidths=0.9, linestyles="--")

    fig.suptitle("Learned S comparison — final layer across CT-MoE variants\n"
                 "blue dashed = uniform baseline (1/15 ≈ 0.067)", y=1.03)
    _save("fig10_S_comparison")


def fig11_ablation_table(results: dict):
    fig, ax = plt.subplots(figsize=(13, 3))
    ax.axis("off")

    cols     = ["Model", "Params (M)", "Code PPL", "Arxiv PPL", "Mixed PPL",
                "Routing Ent", "S Entropy", "S Max", "Route", "Collab"]
    ppl_cols = [2, 3, 4]

    rows = []
    for name, r in results.items():
        cfg  = r["cfg"]
        logs = r["logs"]
        dp   = r["domain_ppl"]
        rows.append([
            DISPLAY_NAMES.get(name, name),
            f"{r['n_params']/1e6:.2f}",
            f"{dp.get('code',  0):.1f}",
            f"{dp.get('arxiv', 0):.1f}",
            f"{dp.get('mixed', 0):.1f}",
            f"{logs['routing_entropy'][-1]:.2f}" if logs["routing_entropy"] else "—",
            f"{logs['s_entropy'][-1]:.3f}"
                if logs.get("s_entropy") and logs["s_entropy"] else "—",
            f"{logs['s_max'][-1]:.4f}"
                if logs.get("s_max") and logs["s_max"] else "—",
            "✗" if (cfg.is_baseline or not cfg.use_graph_routing) else "✓",
            "✗" if (cfg.is_baseline or not cfg.use_graph_collab)  else "✓",
        ])

    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.8)

    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#2C2C2A")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for ri in range(1, len(rows) + 1):
        for ci in range(len(cols)):
            if ri % 2 == 0:
                tbl[ri, ci].set_facecolor("#F1EFE8")
    for ci in ppl_cols:
        vals = []
        for row in rows:
            try:    vals.append(float(row[ci]))
            except: vals.append(float("inf"))
        best = int(np.argmin(vals)) + 1
        tbl[best, ci].set_facecolor("#EAF3DE")
        tbl[best, ci].set_text_props(color="#3B6D11", fontweight="bold")

    ax.set_title("Ablation summary — bold green = best per domain", pad=12)
    _save("fig11_ablation_table")


def make_all_figures(results: dict):
    print(f"\n── Generating figures → ./{FIG_DIR}/ ─────────────────────────")
    fig1_val_ppl(results)
    fig2_train_loss(results)
    fig3_routing_entropy(results)
    fig4_S_heatmaps(results)
    fig5_S_entropy(results)
    fig6_S_max(results)
    fig7_domain_ppl(results)
    fig8_specialization(results)
    fig9_eigenspectrum(results)
    fig10_S_comparison(results)
    fig11_ablation_table(results)
    print(f"\n  Done. All figures saved to ./{FIG_DIR}/")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import platform

    # ── Set EPOCHS = 5 for a ~1.5 hr sanity run, 10 for the full ablation ────
    EPOCHS = 10

    NW = 0 if platform.system() == "Windows" else 2

    print("── Loading data ──────────────────────────────────────────────")
    train_text, val_code_text, val_arxiv_text = load_domain_texts()
    val_mixed_text = (val_code_text[:len(val_code_text)//2] + "\n\n"
                      + val_arxiv_text[:len(val_arxiv_text)//2])

    print("\n── Training tokenizer ────────────────────────────────────────")
    cfg0 = Config()
    sp   = train_spm(train_text, cfg0.vocab_size)

    train_ds = TokenDataset(train_text,     sp, cfg0.seq_len)
    code_ds  = TokenDataset(val_code_text,  sp, cfg0.seq_len)
    arxiv_ds = TokenDataset(val_arxiv_text, sp, cfg0.seq_len)
    mixed_ds = TokenDataset(val_mixed_text, sp, cfg0.seq_len)

    loader_kw    = dict(batch_size=cfg0.batch_size, num_workers=NW,
                        pin_memory=True, persistent_workers=(NW > 0))
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loaders  = {
        "code":  DataLoader(code_ds,  **loader_kw),
        "arxiv": DataLoader(arxiv_ds, **loader_kw),
        "mixed": DataLoader(mixed_ds, **loader_kw),
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"\n  Device : {props.name}  ({props.total_memory/1e9:.1f} GB)")
    else:
        print(f"\n  Device : CPU  (training will be slow)")

    print(f"  Train  : {len(train_ds):,} sequences")
    print(f"  Val    : code={len(code_ds):,}  arxiv={len(arxiv_ds):,}"
          f"  mixed={len(mixed_ds):,}")
    print(f"  Steps/epoch ≈ {len(train_loader):,}  |  "
          f"Total: ~{len(train_loader)*EPOCHS:,} per variant")

    results = run_ablation(train_loader, val_loaders, device, epochs=EPOCHS)

    with open(os.path.join(CKPT_DIR, "all_results.pkl"), "wb") as f:
        pickle.dump(results, f)

    make_all_figures(results)

    print("\n── Final domain PPL ──────────────────────────────────────────")
    print(f"  {'Model':<28} {'Code':>8} {'Arxiv':>8} {'Mixed':>8}  {'Params':>8}")
    print(f"  {'─'*64}")
    for name, r in results.items():
        dp = r["domain_ppl"]
        print(f"  {DISPLAY_NAMES.get(name,name):<28} "
              f"{dp.get('code',0):8.1f} {dp.get('arxiv',0):8.1f}"
              f" {dp.get('mixed',0):8.1f}  {r['n_params']/1e6:7.2f}M")
