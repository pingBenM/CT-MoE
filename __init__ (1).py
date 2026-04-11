"""
CT-MoE configuration.

All hyperparameters live here. The four ablation variants are created
by overriding is_baseline / use_graph_routing / use_graph_collab.
"""

from dataclasses import dataclass


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

    # ── Collaboration topology (S) ─────────────────────────────────────────────
    # routing_scale : multiplier on S column sums used as routing bias.
    #                 Note: row sums of a row-stochastic S are always 1.0 and
    #                 would add a constant offset with no discriminative effect.
    #                 Column sums vary as S learns and carry the real signal.
    # collab_scale  : residual amplitude of message-passing step.
    # s_temp        : softmax temperature for S (lower = sharper distribution).
    # s_lr_scale    : S_raw learning rate multiplier vs base lr.
    #                 S is 256 scalars vs ~65M weights — AdamW will underfit S
    #                 at the base lr without this boost.
    routing_scale : float = 1.5
    collab_scale  : float = 1.0
    s_temp        : float = 1.0
    s_lr_scale    : float = 100.0

    # ── Ablation flags ────────────────────────────────────────────────────────
    # is_baseline=True       → StandardMoE (no S parameter at all)
    # use_graph_routing=False → CT-MoE-NoRouting (message passing only)
    # use_graph_collab=False  → CT-MoE-NoCollab  (routing bias only)
    # both True              → CT-MoE-Full
    is_baseline       : bool = False
    use_graph_routing : bool = True
    use_graph_collab  : bool = True

    # ── Loss ─────────────────────────────────────────────────────────────────
    # Switch-style load balancing applied to all variants for fair comparison.
    lb_weight : float = 0.01

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


# ── Ablation variant registry ─────────────────────────────────────────────────

VARIANTS = [
    ("CT-MoE-Full",      dict()),
    ("StandardMoE",      dict(is_baseline=True)),
    ("CT-MoE-NoCollab",  dict(use_graph_collab=False)),
    ("CT-MoE-NoRouting", dict(use_graph_routing=False)),
]

DISPLAY_NAMES = {
    "CT-MoE-Full":      "CT-MoE (Full)",
    "CT-MoE-NoCollab":  "CT-MoE (No Collab)",
    "CT-MoE-NoRouting": "CT-MoE (No Routing)",
    "StandardMoE":      "Standard MoE",
}

COLORS = {
    "CT-MoE-Full":      "#1D9E75",
    "CT-MoE-NoCollab":  "#378ADD",
    "CT-MoE-NoRouting": "#D85A30",
    "StandardMoE":      "#888780",
}
