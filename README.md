# CT-MoE: Collaborative Topology Mixture-of-Experts

**Paper:** [Learning Expert Collaboration Topology in Mixture-of-Experts Language Models](paper/ct_moe.pdf)

> We show that adding a single learned 16×16 adjacency matrix per MoE layer — 256 scalar parameters — reduces mixed-domain perplexity by **3.6 points absolute** (16.5% relative) over a matched vanilla MoE baseline. A four-condition ablation establishes the gain comes almost entirely from **message passing**, not routing bias.

---

## Key result

| Model | Code PPL | Arxiv PPL | Mixed PPL | Δ Mixed |
|---|---|---|---|---|
| StandardMoE | 25.5 | 19.0 | 22.3 | — |
| CT-MoE (No Collab) | 26.4 | 18.9 | 22.7 | −0.4 |
| CT-MoE (No Routing) | 20.0 | **17.7** | 19.0 | +3.3 |
| **CT-MoE (Full)** | **19.7** | 17.5 | **18.7** | **+3.6** |

65M parameter model. CT-MoE adds **1,536 scalar parameters** across 6 layers (0.0024% overhead).

---

## How it works

Standard MoE selects the top-k experts and combines their outputs independently. CT-MoE adds one step after expert execution: a learned static adjacency matrix **S** governs message passing between the selected experts.

```
Standard MoE:   route → [expert i, expert j] → weighted sum → output
CT-MoE:         route → [expert i, expert j] → message pass via S → weighted sum → output
```

**S** is a single `nn.Parameter(N, N)` per layer, symmetrised, diagonal-masked, and row-normalised. It is optimised end-to-end from the language modelling loss with no auxiliary objectives.

### Why static S, not dynamic?

We tried five dynamic approaches (similarity graphs, output-divergence graphs, learned gates, Hebbian rules, entropy losses). All failed via **geometric saturation**: high-dimensional expert representations collapse to near-uniform pairwise geometry, starving the graph of gradient signal. Static S sidesteps this entirely — `S_raw` receives gradients directly from the task loss. As an extension of this work, we suggest that CT-MoE could be adapted into a dynamic architecture using a hypernetwork. This hypernetwork would generate the adjacency matrix $S$ conditioned on a sequence-level representation, allowing the collaboration topology to adapt to the specific context of the input. We leave this direction for further research.

See Section 4 of the paper for the full failure analysis.

---

## Quickstart

```bash
git clone https://github.com/[your-username]/ct-moe
cd ct-moe
pip install -r requirements.txt

# Full ablation (4 variants × 10 epochs, ~3-4 hr on RTX 4070)
python experiments/run_ablation.py

# Sanity check (5 epochs, ~1.5 hr)
python experiments/run_ablation.py --epochs 5

# Reproduce figures from saved checkpoints
python experiments/make_figures.py --ckpt_dir ct_moe_checkpoints
```

---

## Repository structure

```
ct-moe/
├── src/
│   ├── model.py          # CT-MoE architecture (MoELayer, LanguageModel)
│   ├── data.py           # Dataset loading, tokenisation
│   ├── train.py          # Training loop, evaluation, diagnostics
│   └── config.py         # Config dataclass
├── experiments/
│   ├── run_ablation.py   # Trains all 4 variants, saves checkpoints
│   └── make_figures.py   # Generates all paper figures from checkpoints
├── paper/
│   └── ct_moe.tex        # LaTeX source
├── requirements.txt
└── README.md
```

---

## Citing

```bibtex
@article{maor2026ctmoe,
  title   = {Learning Expert Collaboration Topology in Mixture-of-Experts Language Models},
  author  = {Maor, Ben},
  year    = {2026},
  url     = {https://github.com/[your-username]/ct-moe}
}
```
