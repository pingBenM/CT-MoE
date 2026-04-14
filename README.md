# CT-MoE: Collaborative Topology Mixture-of-Experts

Official code for the paper:

**Learning Expert Collaboration Topology in Mixture-of-Experts Language Models**
Ben Maor — [Article at OpenReview](https://openreview.net/forum?id=LjgzSKpi1A&noteId=LjgzSKpi1A)

---

## What is CT-MoE?

Standard sparse MoE models treat selected expert outputs as independent — they are weighted-summed with no further interaction. CT-MoE adds a single learned static adjacency matrix **S ∈ ℝᴺˣᴺ** per MoE layer that governs message passing between the selected experts after they have each processed their inputs.

**S** is a direct `nn.Parameter` optimised end-to-end from the language modelling objective. It requires no auxiliary loss and adds only N² = 256 (the paper uses N = 16) scalars per layer — less than 0.003% parameter overhead on a 65M baseline.

### Key result

| Model | Code PPL | Arxiv PPL | Mixed PPL |
|---|---|---|---|
| Standard MoE | 25.5 | 19.0 | 22.3 |
| CT-MoE (No Collaboration) | 26.4 | 18.9 | 22.7 |
| CT-MoE (No Routing) | 20.0 | 17.7 | 19.0 |
| **CT-MoE (Full)** | **19.7** | **17.5** | **18.7** |

**3.6-point absolute (16.1% relative) improvement** from 1,536 additional scalar parameters.

The ablation shows the gain comes almost entirely from **message passing**, not routing bias.

---

## Installation

```bash
git clone https://github.com/[YOUR_USERNAME]/ct-moe
cd ct-moe
pip install -r requirements.txt
```

---

## Quick start

```bash
# Full ablation — 4 variants × 10 epochs (~3-4 hr on RTX 4070)
python train.py

# Sanity run — 5 epochs (~1.5 hr)
# Edit EPOCHS = 5 at the bottom of train.py before running
python train.py
```

Figures are saved to `ct_moe_figs/`. Checkpoints are saved to `ct_moe_checkpoints/` after each variant completes — if the run is interrupted, restart and completed variants will be loaded automatically.

---

## Architecture

Each MoE layer contains a learned **N×N adjacency matrix S** derived from a raw parameter `S_raw` through three steps:

1. **Symmetrisation** — `S = (S_raw + S_raw.T) / 2`
2. **Diagonal masking** — `S[i,i] = -inf` (no self-loops)
3. **Softmax normalisation** — row-stochastic distribution over collaboration partners

After expert execution, CT-MoE applies one step of graph message passing over the k selected experts:

```
S_sub = S[selected_experts, selected_experts]   # k×k subgraph
S_sub = renormalise_rows(S_sub)
output = output + collab_scale * (S_sub @ output)
```

The routing bias uses **column sums** of S (not row sums). Row sums of a row-stochastic matrix are always 1.0 — a constant that cancels in softmax. Column sums vary as S learns and provide a real signal.

### Why static S?

Five dynamic topology designs were tested before settling on static S. All failed due to **geometric saturation**: expert representations in high dimensions collapse to near-uniform pairwise geometry, preventing similarity matrices from learning useful structure. Static `S_raw` receives gradients directly from the task loss with no intermediate representation that can saturate.

See Section 4 of the paper for the full negative results record.

---

## Configuration

All hyperparameters live in the `Config` dataclass at the top of `train.py`:

```python
@dataclass
class Config:
    # Architecture
    vocab_size  : int   = 16_000
    seq_len     : int   = 256
    d_model     : int   = 512
    n_layers    : int   = 6
    num_experts : int   = 16
    top_k       : int   = 2

    # CT-MoE topology
    routing_scale : float = 1.5    # column-sum routing bias multiplier
    collab_scale  : float = 1.0    # message passing residual amplitude
    s_temp        : float = 1.0    # softmax temperature for S
    s_lr_scale    : float = 100.0  # S_raw learning rate multiplier

    # Ablation flags
    is_baseline       : bool = False  # True → Standard MoE
    use_graph_routing : bool = True   # enable routing bias
    use_graph_collab  : bool = True   # enable message passing
```

---

## Ablation variants

| Variant key | Routing bias | Message passing |
|---|---|---|
| `CT-MoE-Full` | ✓ | ✓ |
| `CT-MoE-NoCollab` | ✓ | ✗ |
| `CT-MoE-NoRouting` | ✗ | ✓ |
| `StandardMoE` | ✗ | ✗ |

---

## Figures generated

| Figure | Description |
|---|---|
| `fig1_val_ppl.png` | Validation PPL curves — log scale + zoomed linear |
| `fig2_train_loss.png` | Training loss (smoothed) |
| `fig3_routing_entropy.png` | Routing entropy trajectories |
| `fig4_S_*.png` | Learned S heatmaps per layer, per variant |
| `fig5_S_entropy.png` | S row entropy over training |
| `fig6_S_max.png` | Peak collaboration weight over training |
| `fig7_domain_ppl.png` | Domain PPL bar chart |
| `fig8_specialization.png` | Expert domain specialisation scores |
| `fig9_eigenspectrum.png` | S eigenspectrum — final layer |
| `fig10_S_comparison.png` | Side-by-side final-layer S across variants |
| `fig11_ablation_table.png` | Full ablation summary table |

---

## Citation

```bibtex
@article{maor2026ctmoe,
  title   = {Learning Expert Collaboration Topology in
             Mixture-of-Experts Language Models},
  author  = {Maor, Ben},
  journal = {arXiv preprint arXiv:[TBD]},
  year    = {2026}
}
```

---

## License

MIT
