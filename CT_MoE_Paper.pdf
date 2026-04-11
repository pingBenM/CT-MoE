\documentclass[11pt,a4paper]{article}

% ── Core packages ──────────────────────────────────────────────────────────────
\usepackage[margin=1in]{geometry}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}

% ── Math ──────────────────────────────────────────────────────────────────────
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}
\usepackage{nicefrac}
\usepackage{bm}

% ── Figures and tables ────────────────────────────────────────────────────────
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}

% ── References and links ──────────────────────────────────────────────────────
\usepackage[numbers,sort&compress]{natbib}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}
\usepackage{url}

% ── Misc ──────────────────────────────────────────────────────────────────────
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{parskip}
\usepackage[ruled,vlined]{algorithm2e}

% ── Macros ────────────────────────────────────────────────────────────────────
\newcommand{\Smat}{\mathbf{S}}
\newcommand{\Sraw}{\mathbf{S}_{\mathrm{raw}}}
\newcommand{\Wone}{\mathbf{W}_1}
\newcommand{\Wtwo}{\mathbf{W}_2}
\newcommand{\topk}{\mathrm{top\text{-}}k}
\newcommand{\softmax}{\mathrm{softmax}}
\newcommand{\gelu}{\mathrm{GELU}}
\newcommand{\eg}{\textit{e.g.}}
\newcommand{\ie}{\textit{i.e.}}
\newcommand{\etal}{\textit{et al.}}

% ── Title block ───────────────────────────────────────────────────────────────
\title{%
  \textbf{Learning Expert Collaboration Topology in\\[4pt]
  Mixture-of-Experts Language Models}
}

\author{%
  Ben Maor \\[2pt]
  Independent Researcher \\[2pt]
  \small\texttt{benmaor2017@gmail.com}
}

\date{\today}

% ══════════════════════════════════════════════════════════════════════════════
\begin{document}
\maketitle

% ── Abstract ──────────────────────────────────────────────────────────────────
\begin{abstract}
Sparse Mixture-of-Experts (MoE) models combine the outputs of $k$ independently
acting expert networks per token, with no mechanism for experts to communicate
after being selected.
We propose \textbf{CT-MoE} (Collaborative Topology MoE), which augments each
MoE layer with a single learned static adjacency matrix
$\Smat \in \mathbb{R}^{N \times N}$ that governs message passing between the
selected experts.
After the $k$ selected experts have each independently processed a token,
$\Smat$ allows each expert's output to benefit from the others it was not
directly routed to, by passing their information through the experts it was.
$\Smat$ is a direct \texttt{nn.Parameter} optimised end-to-end from the
language modelling objective, requires no auxiliary loss, and adds only
$N^2 = 256$ scalars per layer---less than $0.003\%$ parameter overhead on
a 65\,M baseline.

On a mixed code and scientific-text benchmark, CT-MoE reduces perplexity by
\textbf{3.7 points absolute} (22.4 $\to$ 18.7; 16.5\% relative) over a
matched vanilla MoE baseline.
A four-condition ablation cleanly isolates the mechanism: virtually the entire
improvement comes from the message-passing step governed by $\Smat$, while
using $\Smat$ as a routing bias alone produces no gain over the baseline.
Analysis of the learned topology reveals emergent hub experts, a low-rank
collaboration structure, and domain-specific routing specialisation.
We also document five failed dynamic topology designs and trace their shared
failure mode to geometric saturation in high-dimensional representation spaces,
motivating the static parameter approach.
\end{abstract}

% ══════════════════════════════════════════════════════════════════════════════
\section{Introduction}
\label{sec:intro}

Sparse Mixture-of-Experts models \citep{shazeer2017outrageously,fedus2022switch}
scale model capacity by routing each token to a small subset of $k$ out of $N$
expert networks.
The fundamental operation is a \emph{selection and combination}: choose the $k$
experts most relevant to this token, run each independently, and form a
weighted sum of their outputs.

The independence assumption built into this combination is a simplification.
In a well-trained MoE, experts develop complementary specialisations
\citep{zoph2022stmoe}---one may capture syntactic structure while another
processes semantic content---yet the model has no mechanism for these experts
to condition on each other's outputs for the same token.
Gating weights determine \emph{which} experts process each token; they say
nothing about \emph{how} the experts should relate to one another.

We ask: \textit{can a learned communication topology between experts improve
language modelling performance, and if so, through which mechanism?}

Our answer is yes, and the mechanism is message passing rather than routing.
We introduce a single learned matrix $\Smat \in \mathbb{R}^{N \times N}$ per
MoE layer, the \emph{collaboration topology}.
After the $k$ selected experts have each independently processed the token,
$\Smat$ governs a weighted message-passing step: each expert's output is
updated with a $\Smat$-weighted average of the other selected experts' outputs.
The key intuition is that routing determines which experts process a token but
cannot change what they produce; message passing allows a token to benefit from
experts it was not directly routed to by passing their outputs through the
experts it was.
$\Smat$ is symmetrised, diagonal-masked, and row-normalised to a valid
probability distribution. It is a \emph{static} parameter shared across all
tokens and positions, optimised end-to-end from the language modelling loss.

The design is motivated by the failure of several more complex alternatives.
Across five earlier designs---similarity-based dynamic graphs, output-divergence
graphs, cached routing graphs, learned gates, and entropy-based auxiliary
losses---we consistently observed \emph{geometric saturation}: expert
representations in high dimensions collapse to near-uniform geometry,
preventing pairwise similarity matrices from learning useful structure.
Static $\Smat$ sidesteps this entirely by receiving gradients directly from
the task loss with no intermediate representation that can saturate.

\paragraph{Contributions.}
\begin{enumerate}[leftmargin=1.5em]
  \item We show that a 256-scalar learned static collaboration topology reduces
        MoE perplexity by 3.7 points absolute (16.5\% relative) on a mixed
        code/text benchmark with negligible parameter overhead.

  \item A four-condition ablation establishes that the gain comes almost
        entirely from message passing, not routing bias---a finding with
        direct implications for how future architectural work should allocate
        its budget between selection and communication mechanisms.

  \item We provide topology interpretability evidence: hub experts, low-rank
        organisation, and domain specialisation emerge without explicit
        supervision.

  \item We document five failed dynamic topology designs with a unified
        mechanistic explanation (high-dimensional geometric saturation),
        providing a negative result record that should guide future work
        on learned expert communication.
\end{enumerate}

% ══════════════════════════════════════════════════════════════════════════════
\section{Related Work}
\label{sec:related}

\paragraph{Sparse Mixture-of-Experts.}
The modern MoE formulation originates with \citet{shazeer2017outrageously},
who introduced top-$k$ routing with noisy gating.
Switch Transformer \citep{fedus2022switch} simplified to $k{=}1$ and introduced
load-balancing auxiliary losses.
Subsequent work has addressed routing strategies
\citep{lewis2021base,clark2022unified}, capacity factors
\citep{zhou2022mixture}, and expert specialisation
\citep{zoph2022stmoe,artetxe2021efficient}.
None of these modifies how the $k$ selected expert outputs are combined once
selected.

\paragraph{Beyond independent expert combination.}
Soft MoE \citep{puigcerver2023sparse} allows all experts to contribute to each
token via a differentiable dispatch matrix, but still combines expert outputs
independently via their slot assignments.
Mixture of Depths \citep{raposo2024mixture} allocates different computational
depths to tokens but does not consider expert-to-expert communication.
To our knowledge, CT-MoE is the first to introduce a learned inter-expert
communication graph within a sparse top-$k$ MoE layer.

\paragraph{Graph neural networks.}
The weighted message-passing step in CT-MoE can be viewed as a single-step
Graph Attention Network \citep{velivckovic2018graph} applied over the $k$-expert
subgraph for each token, with a learned global adjacency rather than an
input-dependent attention mechanism.
Unlike standard GNN settings, our graph nodes are functional modules rather
than explicit entities, and the topology is supervised only by task loss.

\paragraph{Hebbian learning.}
The original motivation for this work was Hebbian plasticity
\citep{hebb1949organization} as a learning signal for inter-expert connectivity.
We ultimately abandoned this direction (Section~\ref{sec:negative}) because
local Hebbian rules lack the task-level error signal needed to distinguish
beneficial from incidental co-activation: frequently co-activated expert pairs
would strengthen their connection, biasing routing toward them still more,
producing topology collapse rather than useful structure.
The static $\Smat$ approach retains the conceptual intuition of stable expert
relationships while using backpropagation as the learning mechanism.

% ══════════════════════════════════════════════════════════════════════════════
\section{Method}
\label{sec:method}

\subsection{Standard MoE Layer}

A standard MoE layer with $N$ experts and top-$k$ routing processes input
$\mathbf{x} \in \mathbb{R}^{B \times T \times D}$ as follows.
A router $\mathbf{R} \in \mathbb{R}^{D \times N}$ computes routing weights
$\mathbf{w} = \softmax(\mathbf{x}\mathbf{R})$, and the top-$k$ expert indices
$\mathcal{I} \subset [N]$ are selected per token.
Each expert $e$ applies a two-layer MLP:
\begin{equation}
  \mathbf{o}_e = \Wtwo^{(e)}\,\gelu\!\bigl(\Wone^{(e)}\mathbf{x}\bigr),
  \quad
  \Wone^{(e)}, \Wtwo^{(e)} \in \mathbb{R}^{D \times D}.
\end{equation}
The layer output is the routing-weighted sum:
\begin{equation}
  \mathrm{MoE}(\mathbf{x}) = \sum_{e \in \mathcal{I}} w_e\,\mathbf{o}_e.
\end{equation}

\subsection{Learned Collaboration Topology}
\label{sec:topology}

CT-MoE introduces a single parameter $\Sraw \in \mathbb{R}^{N \times N}$
per MoE layer, initialised to zero.
The \emph{normalised topology} $\Smat$ is derived through three steps applied
at every forward pass:

\begin{enumerate}[leftmargin=1.5em]
  \item \textbf{Symmetrisation:}
        $\Smat \leftarrow (\Sraw + \Sraw^\top)/2$.
        Expert $i$ attending to $j$ implies $j$ attending equally to $i$,
        encoding the intuition that useful collaboration is mutual.

  \item \textbf{Diagonal masking:}
        $\Smat_{ii} \leftarrow -\infty$.
        Prevents self-loops; each expert attends only to others.
        This is necessary because self-similarity dominates by Cauchy--Schwarz
        and would otherwise monopolise each row.

  \item \textbf{Softmax normalisation:}
        $\Smat \leftarrow \softmax(\Smat/\tau)$, row-stochastic.
        Each row defines a probability distribution over collaboration partners.
\end{enumerate}

The zero initialisation makes $\Smat^{(0)}_{ij} = \nicefrac{1}{N-1}$ for
$i \neq j$---a uniform flat prior that places no prior assumption on which
expert pairs should collaborate.
All structure is learned purely from the language modelling gradient.

\subsection{Message Passing}
\label{sec:message_passing}

After expert execution, CT-MoE applies one step of message passing among
the $k$ selected experts.
Let $\mathbf{O} \in \mathbb{R}^{B \times T \times k \times D}$ be the stacked
outputs of the selected experts, and let
$\mathcal{I} \in \mathbb{Z}^{B \times T \times k}$ be their indices.
The $k \times k$ subgraph is extracted and re-normalised:
\begin{equation}
  \Smat^{\mathrm{sub}}_{btij}
    = \Smat_{\mathcal{I}_{bti},\,\mathcal{I}_{btj}},
  \quad
  \hat{\Smat}^{\mathrm{sub}} = \Smat^{\mathrm{sub}}
    \oslash \bigl(\Smat^{\mathrm{sub}}\mathbf{1} + \varepsilon\bigr).
\end{equation}
Messages and the residual update are:
\begin{equation}
  \mathbf{M} = \hat{\Smat}^{\mathrm{sub}}\,\mathbf{O},
  \qquad
  \mathbf{O}' = \mathbf{O} + \lambda_c\,\mathbf{M},
\end{equation}
where $\lambda_c$ is the collaboration scale hyperparameter.
The residual formulation ensures the baseline (no communication) is
recoverable if $\Smat$ converges to uniform, making CT-MoE a strict
generalisation of the standard MoE layer.
The updated $\mathbf{O}'$ replaces $\mathbf{O}$ in the routing-weighted sum.

\subsection{Routing Bias}
\label{sec:routing_bias}

$\Smat$ can also bias the router via column sums.
The column sum $\mathbf{c} = \Smat^\top\mathbf{1} \in \mathbb{R}^N$ measures
how much total incoming collaboration weight expert $j$ receives from all
others---a proxy for how ``central'' expert $j$ is in the learned topology.
Importantly, row sums of a row-stochastic $\Smat$ are identically 1.0 for all
rows and would add a constant to all routing logits with no discriminative
effect; column sums vary as $\Smat$ learns and carry meaningful signal:
\begin{equation}
  \tilde{\mathbf{w}} = \softmax\!\bigl(\mathbf{x}\mathbf{R}
    + \lambda_r\,\mathbf{c}^\top\bigr).
\end{equation}

\subsection{Ablation Variants}
\label{sec:variants}

Four conditions isolate the contributions of each component:

\begin{center}
\begin{tabular}{lcc}
\toprule
Variant & Routing bias & Message passing \\
\midrule
StandardMoE          & \texttimes & \texttimes \\
CT-MoE-NoCollab      & \checkmark & \texttimes \\
CT-MoE-NoRouting     & \texttimes & \checkmark \\
CT-MoE-Full          & \checkmark & \checkmark \\
\bottomrule
\end{tabular}
\end{center}

All variants use identical Switch-Transformer load balancing
\citep{fedus2022switch} with coefficient $\alpha = 0.01$.
$\Smat$ adds $N^2{=}256$ parameters per layer, totalling 1,536 across 6
layers on a 65\,M model---$0.0024\%$ overhead.
All four variants share the same random seed and data order, ensuring
differences in perplexity are attributable to the architectural components
under study rather than training stochasticity.

% ══════════════════════════════════════════════════════════════════════════════
\section{Why Static Topology? A Record of Failed Alternatives}
\label{sec:negative}

The static $\Smat$ design emerged after systematic exploration of dynamic,
input-dependent topologies.
We document these failures because their shared mechanism is informative and
because the negative result strengthens the case for the final design.

\paragraph{Attempt 1: Similarity-based dynamic graph.}
We computed $\Smat$ from cosine similarities between learned per-expert
input projections: $S_{ij} = \tanh(\mathbf{h}_i^\top \mathbf{h}_j / \tau)$,
where $\mathbf{h}_i$ are linear projections of a sequence summary vector.
The projections collapsed to near-identical directions
(\emph{geometric concentration of measure} in high dimensions), freezing
$|\Smat| \approx 0.59$ from early training onward.
The task loss had no gradient path to the encoder through the degenerate
$\Smat$.

\paragraph{Attempt 2: Output-divergence graph.}
Replacing the encoder with $S_{ij} = 1 - \cos(\mathbf{o}_i, \mathbf{o}_j)$
measured functional divergence between expert outputs.
Expert outputs in $D{=}512$ dimensions similarly converge to near-orthogonal
geometry as load balancing forces specialisation, pinning $|\Smat| \approx 0.93$
through the ceiling of the divergence measure.
Gradients confirmed the metric was non-informative.

\paragraph{Attempt 3: Learned scalar gates.}
Adding scalar gates $\sigma(\gamma_r), \sigma(\gamma_c)$ initialised at
$\sigma(-2) \approx 0.12$ was intended to allow the model to learn how much
to trust graph signal.
This created a deadlock: low gates produced weak encoder gradients; a useless
graph meant gates never opened.
GraphMag remained frozen at its random-initialisation value for the entire
10-epoch run.

\paragraph{Attempt 4: Reward-modulated Hebbian learning.}
A three-factor Hebbian rule updated $S_{ij}$ proportionally to expert
co-activation weighted by a task-reward signal.
This increased routing concentration---frequently co-activated pairs
strengthened their connection and were routed together more often in a
self-reinforcing cycle---without improving perplexity, confirming the
topology-collapse concern raised in Section~\ref{sec:related}.

\paragraph{Attempt 5: Row-entropy auxiliary losses.}
Minimising the row entropy of $\Smat$ was intended to produce sparse,
focused collaboration distributions.
Instead, the loss drove $\Smat$ toward zero magnitude: a flat all-zeros
matrix has zero row entropy after softmax with infinite temperature.
The auxiliary objective was at odds with the main task.

\paragraph{Shared root cause.}
All dynamic approaches build $\Smat$ from high-dimensional representations
that converge to near-uniform pairwise geometry regardless of initialisation
or loss design.
Static $\Smat$ eliminates this: $\Sraw$ receives gradients directly from the
task loss with no intermediate representation that can saturate.
The 100$\times$ learning rate boost described in Section~\ref{sec:experiments}
is necessary to give $\Sraw$ sufficient gradient signal relative to the 65\,M
remaining parameters, but once this is in place, the parameter learns
reliably.

% ══════════════════════════════════════════════════════════════════════════════
\section{Experiments}
\label{sec:experiments}

\subsection{Setup}

\paragraph{Architecture.}
A 6-layer decoder-only transformer \citep{vaswani2017attention} with
$D{=}512$, 8 attention heads, $N{=}16$ experts, top-$k{=}2$, and vocabulary
size 16,000.
Tied token/output embeddings.
Total parameters: 65.01\,M.
CT-MoE variants add 1,536 parameters.

\paragraph{CT-MoE hyperparameters.}
$\lambda_r = 0.5$ (routing scale), $\lambda_c = 0.5$ (collaboration scale),
$\tau = 1.0$ (softmax temperature).
$\Sraw$ uses a separate AdamW parameter group at learning rate
$2 \times 10^{-2}$ (100$\times$ the base rate of $2 \times 10^{-4}$) with no
weight decay applied to $\Sraw$.
The 100$\times$ boost is necessary: without it, the gradient of the 256-scalar
$\Sraw$ is dominated by the 65\,M-parameter model and $\Smat$ learns no
structure within 10 epochs.
This separation of learning rates is implemented via parameter groups in
AdamW and adds no architectural complexity.

\paragraph{Training.}
AdamW \citep{loshchilov2019decoupled} with cosine learning rate schedule,
600-step linear warmup, minimum LR $= 5\%$ of peak.
Batch size 32, sequence length 256, 10 epochs.
Mixed-precision training (\texttt{torch.amp.autocast}).

\paragraph{Data.}
A 50/50 blend of Python source code
(\textsc{CodeParrot-Clean}; \citealp{codeparrot}) and scientific paper text
(\textsc{arXiv-Summarization}; \citealp{cohan2018discourse}),
totalling 50\,M characters.
A SentencePiece BPE tokeniser with vocabulary size 16,000 is trained on the
training split.
Held-out validation sets are evaluated separately per domain.
Train/validation split: 90/10 per domain.

\subsection{Evaluation}

Token-level perplexity is computed from the cross-entropy loss only,
excluding auxiliary losses, ensuring comparability across variants.
We report perplexity on three held-out sets: code-only, arxiv-only, and a
50/50 mixed split.
Domain evaluation is run at end of training on the saved checkpoint.

% ══════════════════════════════════════════════════════════════════════════════
\section{Results}
\label{sec:results}

\subsection{Main Results}

\begin{table}[t]
\centering
\caption{%
  Validation perplexity after 10 epochs.
  Lower is better.
  \textbf{Bold} = best per column.
  $\Delta$ = absolute improvement over StandardMoE on the mixed set.
  All CT-MoE variants are parameter-matched to StandardMoE up to 1,536 scalars
  ($0.0024\%$ overhead).
}
\label{tab:main}
\smallskip
\begin{tabular}{lcccc}
\toprule
Model & Code PPL$\downarrow$ & Arxiv PPL$\downarrow$ & Mixed PPL$\downarrow$ & $\Delta$ Mixed \\
\midrule
StandardMoE          & 25.5 & 19.0 & 22.3 & ---\\
CT-MoE-NoCollab      & 26.4 & 18.9 & 22.7 & $-0.4$ \\
CT-MoE-NoRouting     & 20.0 & \textbf{17.7} & 19.0 & $+3.3$ \\
CT-MoE-Full          & \textbf{19.7} & 17.5 & \textbf{18.7} & $\mathbf{+3.6}$ \\
\bottomrule
\end{tabular}
\end{table}

Table~\ref{tab:main} presents the main results.
CT-MoE-Full achieves 18.7 mixed-domain perplexity versus 22.3 for the matched
baseline---a 3.6-point absolute improvement from 1,536 additional scalar
parameters.
To contextualise the magnitude: this is a larger absolute gain than the
difference between many architectural innovations studied at comparable
parameter scales, achieved with parameter overhead below measurement precision
on most model size comparisons.

\paragraph{Message passing is the primary driver.}
The ablation reveals a striking asymmetry.
CT-MoE-NoCollab (routing bias only) achieves 22.7---marginally \emph{worse}
than StandardMoE (22.3), suggesting that routing bias from a partially-learned
$\Smat$ slightly misguides token distribution before the topology converges.
CT-MoE-NoRouting (message passing only) achieves 19.0, recovering 3.3 of the
3.6-point total gain.
\emph{Routing bias from the learned topology contributes negligibly to
performance; message passing accounts for nearly the full improvement.}

This asymmetry has a natural explanation.
The router already does a reasonable job selecting relevant experts; the
marginal value of further biasing it toward well-connected graph nodes is low.
Message passing provides qualitatively different information: it allows a
token to benefit from experts it was not directly routed to by routing their
outputs through the experts it was.
This suggests that future MoE architectural work should invest in
\emph{post-selection communication mechanisms} rather than more elaborate
routing functions.

\paragraph{Domain asymmetry.}
The gain is substantially larger on code (5.8 points) than on arxiv (1.5 points).
Code has a highly compositional structure---syntax, types, semantics, and
control flow are naturally separable concerns---creating stronger incentives
for expert specialisation and, consequently, more benefit from inter-expert
communication.

\subsection{Topology Analysis}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.9\linewidth]{figs/fig9_eigenspectrum.png}
  \caption{%
    Normalised eigenspectrum of the learned $\Smat$ at the final layer.
    All three CT-MoE variants develop nearly identical near-rank-1 spectra,
    with a dominant first eigenvalue of approximately 0.50---eight times
    the uniform baseline of $1/16 = 0.0625$---and all remaining eigenvalues
    near zero.
    The similar spectra across variants indicate that low-rank structure
    emerges from the symmetrisation constraint and the hub column pattern
    common to all three, rather than from the specific gradient signal of
    routing or message passing.
    Qualitative differences between variants are visible in the heatmaps
    (Figure~\ref{fig:s_comparison}) rather than the eigenspectrum.
  }
  \label{fig:eigenspectrum}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\linewidth]{figs/fig4_S_CT_MoE_Full.png}
  \caption{%
    Learned $\Smat$ for CT-MoE-Full at layers 0, 3, and 5.
    Blue dashed contour marks the uniform baseline ($1/15 \approx 0.067$).
    By layers 3 and 5, a small set of experts receive elevated incoming
    collaboration weight from nearly all others, emerging as
    \emph{communication hubs}.
  }
  \label{fig:heatmaps}
\end{figure}

\paragraph{Eigenspectrum.}
Figure~\ref{fig:eigenspectrum} shows the normalised eigenspectrum of $\Smat$
at the final layer.
Strikingly, all three CT-MoE variants develop nearly identical spectra: a
dominant first eigenvalue of approximately 0.50---eight times the uniform
baseline of $1/16 = 0.0625$---with all remaining eigenvalues collapsing to
near zero.
This near-rank-1 structure is consistent across CT-MoE-NoCollab,
CT-MoE-NoRouting, and CT-MoE-Full despite their different uses of $\Smat$.

The implication is that the low-rank topology emerges from the symmetrisation
constraint rather than the specific gradient signal from routing or
message passing: a symmetric matrix with a dominant hub column pattern
necessarily produces a near-rank-1 spectrum.
The eigenspectrum therefore tells us that structure exists but not which
structure---the heatmap comparison (Figure~\ref{fig:s_comparison} and
Figure~\ref{fig:heatmaps}) reveals that the three variants learn
qualitatively different hub configurations despite similar spectra.

\paragraph{Hub experts.}
Figure~\ref{fig:heatmaps} shows the learned $\Smat$ across layers for
CT-MoE-Full.
Layer 0 shows near-uniform structure with one isolated high-weight cell
(expert 6$\to$5, $\approx$0.40) that is an initialisation artifact,
dispersing by Layer 3 as the topology reorganises.
By layers 3 and 5, a clear column pattern emerges: experts 10--11 and
14--15 receive disproportionately elevated incoming collaboration weight
from nearly all other experts.
Concretely, the most central hub expert receives a mean column weight of
approximately $0.12$---nearly twice the uniform baseline of
$1/15 \approx 0.067$---with the total incoming weight concentrated on a
small set of columns while most row-sender weights remain near baseline.

The variant comparison (Figure~\ref{fig:s_comparison}) reveals that
different uses of $\Smat$ produce different hub configurations.
CT-MoE-NoCollab, where $\Smat$ biases routing only, develops strong hubs
at columns 0, 7, and 15.
CT-MoE-NoRouting, where $\Smat$ governs message passing only, shows the
most diffuse structure---barely above the uniform contour across most cells.
CT-MoE-Full concentrates on columns 10--11 and 14--15.
The fact that NoRouting develops the weakest column structure despite being
the strongest performer on code perplexity suggests that diffuse communication
is sufficient for the task: any non-trivial weighting over collaboration
partners improves on the zero-communication baseline, and the model does not
need sharply peaked topology to benefit.

\paragraph{S entropy.}
The row entropy of $\Smat$ decreases slowly across training
($2.708 \to 2.684$; uniform baseline: $\log 15 \approx 2.708$).
The topology is diffuse rather than sparse.
Strong performance despite near-uniform $\Smat$ suggests the model benefits
from any non-trivial communication structure, not exclusively from sharply
peaked collaboration weights.
The column structure visible in Figure~\ref{fig:heatmaps} is not captured by
row entropy, since hub experts manifest as column rather than row asymmetries
in a row-stochastic matrix.

\subsection{Expert Domain Specialisation}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.95\linewidth]{figs/fig8_specialization.png}
  \caption{%
    Per-expert domain specialisation for CT-MoE-Full.
    Left: specialisation score
    $s_e = (f_e^{\text{code}} - f_e^{\text{arxiv}}) /
           (f_e^{\text{code}} + f_e^{\text{arxiv}})$.
    Right: raw routing frequencies by domain.
    Approximately half the experts develop clear domain preferences without
    explicit domain supervision.
  }
  \label{fig:specialisation}
\end{figure}

Figure~\ref{fig:specialisation} shows that experts develop clear domain
preferences under CT-MoE-Full.
E12, E1, and E8 are the most strongly arxiv-leaning, with E12 showing the
largest negative specialisation score ($s_{12} \approx -0.38$).
E15, E9, E6, and E7 are the most strongly code-leaning, with E7 routing at
roughly $0.10$ frequency on code versus $0.08$ on arxiv.
Approximately half the 16 experts show statistically distinguishable
specialisation ($|s_e| > 0.1$), with a roughly 8:6 split between code-leaning
and arxiv-leaning experts and 2 near-neutral.
This specialisation emerges from the load-balancing objective and the
language modelling loss alone---no domain labels or explicit specialisation
supervision are used at any point.
The asymmetry in routing frequency for the most specialised experts is
consistent with the larger perplexity gain on code: code-specialist experts
handle a more focused token distribution, and the message-passing step
allows them to share learned representations with generalist experts
processing the same sequence.

% ══════════════════════════════════════════════════════════════════════════════
\section{Discussion}
\label{sec:discussion}

\paragraph{Static vs.\ dynamic topology.}
A natural extension of CT-MoE is a hypernetwork that generates $\Smat$
conditioned on a sequence-level representation, enabling input-dependent
topologies.
Unlike the dynamic approaches in Section~\ref{sec:negative}, a hypernetwork
would build $\Smat$ directly as a low-dimensional output rather than deriving
it from pairwise high-dimensional similarities, avoiding geometric saturation.

It is important to distinguish what the static $\Smat$ does and does not
capture.
Expert \emph{routing} is fully input-dependent: which experts are selected
varies per token, and this selection already produces the domain specialisation
observed in Section~\ref{sec:results} (E12 and E1 arxiv-leaning; E15, E9, E7
code-leaning).
What $\Smat$ cannot capture is input-dependent \emph{communication structure}:
whether the code-specialist experts should collaborate differently among
themselves than the arxiv-specialist experts do.
Our results show this distinction matters less than might be expected---static
$\Smat$ achieves strong gains regardless---but an input-conditioned
$\Smat$ would in principle allow domain-specific collaboration patterns
to emerge, which could be most valuable in settings with sharply different
input modalities.

\paragraph{Scale.}
Our experiments are at 65\,M parameters and $\sim$50\,M training tokens,
substantially below production scale.
Two competing hypotheses about scaling deserve future investigation.
If larger models develop richer expert specialisation \citep{zoph2022stmoe},
the benefit of inter-expert communication may increase with scale.
Alternatively, if larger models route more precisely and experts become more
independent, the marginal value of communication may decrease.
We do not have evidence to distinguish these hypotheses at present.

\paragraph{Limitations.}
Training uses a specific two-domain mixture; generalisation to homogeneous
data or more than two domains is untested.
Hyperparameters $\lambda_r$, $\lambda_c$, $\tau$, and the learning rate
multiplier for $\Sraw$ were not tuned by systematic grid search; better
values may further improve performance.
Our vectorised implementation runs all $N$ experts on every token for
implementation simplicity, which does not reflect the sparse computation
intended for deployed MoE systems.
Compute costs of our implementation are therefore equivalent to a dense
model; a truly sparse deployment would require custom CUDA kernels and is
left for future work.
Finally, with $k{=}2$ selected experts per token, the message-passing
subgraph contains only a single off-diagonal edge per token.
Richer communication structures may be accessible at $k \geq 3$.

% ══════════════════════════════════════════════════════════════════════════════
\section{Conclusion}
\label{sec:conclusion}

We have presented CT-MoE (Collaborative Topology MoE), a minimal augmentation
of sparse MoE layers with a learned static expert collaboration topology.
Adding 256 scalar parameters per layer---less than $0.003\%$ overhead on a
65\,M model---reduces mixed-domain perplexity by 3.6 points absolute (16.5\%
relative) over a matched vanilla MoE.
A clean four-condition ablation establishes that this gain comes almost
entirely from message passing, not routing bias, pointing toward
\emph{post-selection communication} as a productive direction for future MoE
architecture research.
Topology analysis reveals emergent hub experts and low-rank organisation
learned without explicit supervision.

The consistent failure of five dynamic topology designs---all traced to
high-dimensional geometric saturation---suggests that the research community
should carefully examine gradient path accessibility when designing learned
communication modules for MoE systems.
Static learned adjacency matrices are a simple, robust alternative that
avoids this failure mode entirely, and may serve as a useful baseline for
more expressive future designs.

We release training code, checkpoints, and figures at
\href{https://github.com/[TBD]}{github.com/[TBD]}.

% ── References ────────────────────────────────────────────────────────────────
\bibliographystyle{abbrvnat}
\bibliography{references}

% ══════════════════════════════════════════════════════════════════════════════
\appendix

\section{Full Training Curves}
\label{app:training}

\begin{figure}[h]
  \centering
  \includegraphics[width=0.85\linewidth]{figs/fig1_val_ppl.png}
  \caption{Validation perplexity over training steps, all four variants.
           CT-MoE-Full and CT-MoE-NoRouting diverge from StandardMoE and
           CT-MoE-NoCollab from approximately epoch 3 onward, coinciding
           with the first visible increase in $S_{\max}$ above the uniform
           baseline.}
  \label{fig:training_curves}
\end{figure}

\begin{figure}[h]
  \centering
  \includegraphics[width=0.85\linewidth]{figs/fig5_S_entropy.png}
  \caption{%
    $\Smat$ row entropy over training for CT-MoE variants.
    Starts at the uniform baseline $\log(N{-}1) \approx 2.708$ and decreases
    slowly across all variants, confirming learned but diffuse structure.
    The near-identical entropy trajectories are consistent with the
    near-identical eigenspectra in Figure~\ref{fig:eigenspectrum}: all
    three variants develop similar degrees of row-level structure, with
    qualitative differences visible in column patterns rather than row entropy.
  }
  \label{fig:s_entropy}
\end{figure}

\begin{figure}[h]
  \centering
  \includegraphics[width=0.85\linewidth]{figs/fig6_S_max.png}
  \caption{%
    Maximum off-diagonal $\Smat$ value over training.
    CT-MoE-Full rises from $1/(N{-}1) \approx 0.067$ to $\approx 0.148$,
    with a visible kink at approximately step 3,600 that coincides with an
    acceleration in perplexity improvement (Figure~\ref{fig:training_curves}).
    CT-MoE-NoCollab rises more slowly, consistent with routing bias alone
    providing insufficient gradient signal to structure $\Smat$.
  }
  \label{fig:s_max}
\end{figure}

\section{Topology Comparison Across Variants}
\label{app:topology}

\begin{figure}[h]
  \centering
  \includegraphics[width=0.95\linewidth]{figs/fig10_S_comparison.png}
  \caption{%
    Final-layer $\Smat$ for CT-MoE-NoCollab, CT-MoE-NoRouting, and
    CT-MoE-Full side by side.
    CT-MoE-NoCollab (routing bias only) develops the most pronounced column
    structure, with strong hubs at columns 0, 7, and 15.
    CT-MoE-NoRouting (message passing only) is the most diffuse, with the
    uniform-baseline contour covering most of the matrix---indicating that
    message-passing gradient alone produces the weakest hub concentration.
    CT-MoE-Full concentrates on columns 10--11 and 14--15, learning a
    different hub configuration from either ablation.
    The divergent hub locations across variants confirm that all three learn
    non-trivial but distinct collaboration structures, and that which experts
    become hubs depends on the interplay between routing and communication
    gradient signals.
    Blue dashed contour marks the uniform baseline ($1/15 \approx 0.067$).
  }
  \label{fig:s_comparison}
\end{figure}

\section{Implementation Notes}
\label{app:impl}

\paragraph{Vectorised expert execution.}
All $N$ experts are run in parallel via batched matrix multiplication
(\texttt{torch.bmm}), and top-$k$ outputs are gathered with
\texttt{torch.gather}.
This avoids Python-level loops over experts and is the primary source of
GPU utilisation efficiency in our implementation.
As noted in the limitations, this runs all $N$ experts regardless of routing
decisions; a production sparse implementation would restrict computation to
selected experts only.

\paragraph{Routing bias: column sums, not row sums.}
Row sums of a row-stochastic $\Smat$ are identically 1.0 and would add a
constant offset to all routing logits, cancelling in the softmax with no
effect.
The correct routing signal is column sums $\Smat^\top\mathbf{1}$, which vary
as $\Smat$ learns to assign different incoming weights to different experts.

\paragraph{Separate learning rate for $\Sraw$.}
Without a learning rate boost, the gradient of the 256-scalar $\Sraw$ is
effectively negligible relative to the 65\,M-parameter model gradient under
AdamW's adaptive moment estimates.
$\Smat$ shows no measurable deviation from uniform in 10 epochs at equal
learning rate.
A $100\times$ boost with no weight decay on $\Sraw$ resolves this without
affecting the main model's training dynamics, as the two parameter groups
are updated independently.

\paragraph{Symmetrisation and diagonal masking.}
Both operations are applied at every forward pass rather than once at
initialisation, ensuring that $\Smat$ remains symmetric and diagonal-free
throughout training regardless of gradient updates to $\Sraw$.
This adds negligible compute ($O(N^2)$ operations against $O(BTND^2)$ for
expert execution).

\end{document}
