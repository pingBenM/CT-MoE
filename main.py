"""
CT-MoE model architecture.

Core contribution: MoELayer with a learned static adjacency matrix S.

S is an nn.Parameter(N, N) per layer, optimised end-to-end from the LM loss.
  - Symmetrised:      S[i,j] == S[j,i]  (mutual collaboration)
  - Diagonal-masked:  S[i,i] = 0        (no self-loops)
  - Row-stochastic:   rows sum to 1     (valid probability distribution)

Routing bias uses COLUMN sums of S, not row sums.
  - Row sums are always 1.0 by construction → constant offset → useless.
  - Column sums vary as S learns → measures each expert's incoming attention.

Message passing uses the K×K subgraph for the selected experts, re-normalised
after extraction. The residual formulation means CT-MoE degenerates exactly
to StandardMoE when S is uniform.

S_raw is given a separate AdamW param group at 100× the base learning rate.
Without this, S is a 256-scalar parameter against 65M others — AdamW's
adaptive moments prevent it from learning any structure at the base lr.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


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

        def rs(t):
            return t.view(B, T, self.n_head, hd).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            rs(q), rs(k), rs(v),
            is_causal=True,
            dropout_p=self.p if self.training else 0.0,
        )
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MoELayer(nn.Module):
    """
    CT-MoE MoE layer — supports all four ablation variants via Config flags.

    Variants:
      is_baseline=True         → StandardMoE    (no S, no message passing)
      use_graph_collab=False   → CT-MoE-NoCollab  (routing bias only)
      use_graph_routing=False  → CT-MoE-NoRouting (message passing only)
      both True (default)      → CT-MoE-Full

    All variants use Switch-style load balancing with coefficient lb_weight.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        N, D = cfg.num_experts, cfg.d_model

        # Expert weights — shared across all variants
        self.W1     = nn.Parameter(torch.randn(N, D, D) / D ** 0.5)
        self.W2     = nn.Parameter(torch.randn(N, D, D) / D ** 0.5)
        self.router = nn.Linear(D, N, bias=False)
        self.register_buffer("usage", torch.zeros(N))

        # Collaboration topology — only instantiated for CT-MoE variants
        self.has_graph = not cfg.is_baseline
        if self.has_graph:
            # Zero init → softmax gives uniform 1/(N-1) after diag mask.
            # All structure learned purely from LM gradient.
            self.S_raw = nn.Parameter(torch.zeros(N, N))

        # Diagnostics (updated each forward, logged by training loop)
        self.last_s_entropy = 0.0
        self.last_s_max     = 0.0

    # ── Topology ──────────────────────────────────────────────────────────────

    def get_S(self) -> torch.Tensor:
        """
        Derives the normalised (N, N) adjacency from S_raw.

        Steps:
          1. Symmetrise   — S = (S_raw + S_raw^T) / 2
          2. Mask diagonal — S[i,i] = -inf before softmax
          3. Softmax / temp — row-stochastic probability distribution

        Called every forward pass. O(N^2) = 256 ops; negligible vs expert fwd.
        """
        N   = self.cfg.num_experts
        dev = self.S_raw.device

        S = (self.S_raw + self.S_raw.T) / 2
        S = S.masked_fill(torch.eye(N, dtype=torch.bool, device=dev), float("-inf"))
        S = torch.softmax(S / self.cfg.s_temp, dim=-1)

        with torch.no_grad():
            ent = -(S * (S + 1e-9).log()).sum(dim=-1).mean()
            self.last_s_entropy = ent.item()
            S_nd = S.clone().fill_diagonal_(0)
            self.last_s_max = S_nd.max().item()

        return S

    # ── Load balancing ────────────────────────────────────────────────────────

    def lb_loss(self, weights: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
        """Switch-Transformer load balancing: N · Σᵢ fᵢ · pᵢ"""
        N = self.cfg.num_experts
        f = F.one_hot(topk_idx, N).float().sum(dim=-2).mean(dim=(0, 1))
        p = weights.mean(dim=(0, 1))
        return N * (f * p).sum()

    # ── Routing ───────────────────────────────────────────────────────────────

    def route(self, x: torch.Tensor, S) -> tuple:
        """
        Top-k routing with optional column-sum bias from S.

        Column sums of S (not row sums) give the routing bias.
        Row sums = 1 always (row-stochastic) → useless constant offset.
        Column sums vary as S learns → proxy for expert centrality.
        """
        B, T, D = x.shape
        logits   = self.router(x.view(B * T, D)).view(B, T, self.cfg.num_experts)

        if self.has_graph and self.cfg.use_graph_routing and S is not None:
            col_sums = S.sum(dim=0)  # (N,) — varies with S
            logits   = logits + self.cfg.routing_scale * col_sums

        weights            = torch.softmax(logits, dim=-1)
        topk_val, topk_idx = torch.topk(weights, self.cfg.top_k, dim=-1)

        if self.training:
            self.usage.scatter_add_(
                0,
                topk_idx.flatten(),
                torch.ones(topk_idx.numel(), device=topk_idx.device,
                           dtype=self.usage.dtype),
            )

        return weights, topk_val, topk_idx

    # ── Expert execution ──────────────────────────────────────────────────────

    def expert_fwd(self, x: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
        """
        Vectorised expert execution via batched matmul.
        Runs all N experts in parallel, then gathers the top-k outputs.
        O(N·BT·D²) — avoids Python-level expert loops.
        """
        B, T, D = x.shape
        K, N    = self.cfg.top_k, self.cfg.num_experts

        xf      = x.view(B * T, D)
        xf_exp  = xf.unsqueeze(0).expand(N, -1, -1).contiguous()  # (N, BT, D)
        h       = F.gelu(torch.bmm(xf_exp, self.W1))               # (N, BT, D)
        out_all = torch.bmm(h, self.W2)                            # (N, BT, D)

        out_t     = out_all.permute(1, 0, 2)                       # (BT, N, D)
        idx       = topk_idx.view(B * T, K)
        out_stack = torch.gather(
            out_t, 1, idx.unsqueeze(-1).expand(B * T, K, D)
        )  # (BT, K, D)

        return out_stack.view(B, T, K, D)

    # ── Message passing ───────────────────────────────────────────────────────

    def interact(self, out_stack: torch.Tensor,
                 topk_idx: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """
        One step of S-weighted message passing over the K selected experts.

        For each token, extracts the K×K subgraph from S, re-normalises rows
        (subgraph extraction breaks row-stochastic property), then computes
        a weighted average of expert outputs as the message.

        Residual connection: output = expert_out + collab_scale * message.
        When S is uniform, this reduces to output + collab_scale * mean(outputs),
        which approaches the identity as collab_scale → 0.
        """
        B, T, K, D = out_stack.shape

        # Extract K×K subgraph via advanced indexing
        ei    = topk_idx.unsqueeze(-1).expand(B, T, K, K)  # row indices
        ej    = topk_idx.unsqueeze(-2).expand(B, T, K, K)  # col indices
        S_sub = S[ei, ej]                                   # (B, T, K, K)

        # Re-normalise after extraction
        S_sub = S_sub / (S_sub.sum(dim=-1, keepdim=True) + 1e-9)

        msg = torch.matmul(S_sub, out_stack)  # (B, T, K, D)
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
    """
    Decoder-only transformer with CT-MoE layers.

    Architecture: 6 layers, D=512, 8 heads, N=16 experts, top-k=2.
    Total parameters: ~65M. CT-MoE variants add 1,536 scalars (0.0024%).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg    = cfg
        self.emb    = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos    = nn.Embedding(cfg.seq_len,    cfg.d_model)
        self.drop   = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f   = nn.LayerNorm(cfg.d_model)
        self.head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.emb.weight  # weight tying
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor, targets=None):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        x    = self.drop(self.emb(idx) + self.pos(pos))
        aux  = 0.0
        for block in self.blocks:
            x, a = block(x)
            aux += a
        logits = self.head(self.ln_f(x))
        ce = total = None
        if targets is not None:
            ce    = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            total = ce + aux / self.cfg.n_layers
        return logits, total, ce

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def zero_usage(self):
        for b in self.blocks:
            b.moe.usage.zero_()

    def make_optimizer(self, cfg: Config) -> torch.optim.Optimizer:
        """
        Separate AdamW param groups: S_raw at 100× base lr, no weight decay.
        Without the lr boost, S is a 256-scalar parameter against 65M others
        and learns no structure within 10 epochs.
        """
        s_params     = [b.moe.S_raw for b in self.blocks if hasattr(b.moe, "S_raw")]
        other_params = [p for n, p in self.named_parameters()
                        if not any(p is sp for sp in s_params)]
        param_groups = [{"params": other_params, "lr": cfg.lr}]
        if s_params:
            param_groups.append({
                "params":       s_params,
                "lr":           cfg.lr * cfg.s_lr_scale,
                "weight_decay": 0.0,
            })
        return torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    def get_all_S(self) -> list:
        """Returns list of (N, N) numpy arrays, one per layer."""
        return [b.moe.get_S_numpy() for b in self.blocks]

    @torch.no_grad()
    def get_routing_freq(self, loader, device, n_batches: int = 100):
        was_training = self.training
        self.train()
        self.zero_usage()
        for i, (x, _) in enumerate(loader):
            if i >= n_batches:
                break
            self(x.to(device))
        freq = torch.zeros(self.cfg.num_experts)
        for b in self.blocks:
            freq += b.moe.usage.cpu()
        self.zero_usage()
        if not was_training:
            self.eval()
        return freq / (freq.sum() + 1e-9)
