# DITTO-AV: Graph-Based DITTO for Autonomous Driving Behavioral Simulation

Extension of [DITTO (Offline Imitation Learning with World Models)](https://arxiv.org/abs/2302.03086) to autonomous driving using graph-based scene representations.

## Repository Structure

```
src/
├── (original DITTO)         # Original pixel-based DITTO for Atari (reference)
│   ├── models/              # CNN encoder, RSSM, pixel decoder, discrete actor-critic
│   ├── trainers/            # World model & actor-critic trainers
│   ├── data/                # D4RL / image datasets
│   └── config/              # Atari configs
│
└── graph_ditto/             # NEW: Graph-based DITTO for driving
    ├── models/              # Scene encoder (attention GNN), RSSM, scene decoder,
    │                        #   graph world model, continuous actor-critic
    ├── data/                # Scene graph builder, driving dataset, synthetic generator
    ├── config/              # Option 1 & Option 2 YAML configs + config builder
    ├── trainers/            # WM trainer, ego trainer (Opt 1), multi-agent trainer (Opt 2)
    ├── train_option1.py     # Entry point: ego-only DITTO
    └── train_option2.py     # Entry point: multi-agent shared-policy DITTO
```

## Two Options

### Option 1: Ego-Only Policy
A single ego-agent learns to imitate expert driving via DITTO's latent divergence reward in a graph-based world model. Direct analog of original DITTO with the CNN encoder replaced by an attention-based scene graph encoder.

```bash
cd src
python -m graph_ditto.train_option1 --config graph_ditto/config/option1_config.yaml
```

### Option 2: Multi-Agent Shared Policy
All agents share one policy network. Per-agent GNN embeddings + RSSM latent → per-agent actions. An `AgentStateUpdater` maintains per-agent embeddings during dream rollouts. DITTO reward is scene-level: `max_cos(h_policy, h_expert)`.

```bash
cd src
python -m graph_ditto.train_option2 --config graph_ditto/config/option2_config.yaml
```

### Staged Training

Both options follow a two-stage pipeline:
1. **World Model Training**: Train `GraphWorldModel` (SceneEncoder + RSSM + SceneDecoder) on expert driving data using ELBO objective.
2. **Policy Training**: Freeze WM, featurize expert data into RSSM latents, train policy by unrolling in WM with `max_cos` reward.

You can run stages separately:
```bash
# Stage 1 only
python -m graph_ditto.train_option1 --stage wm

# Stage 2 only (with pre-trained WM)
python -m graph_ditto.train_option1 --stage policy --wm-checkpoint checkpoints/option1/wm/wm_step100000.pt
```

## Key Architecture Decisions

- **Scene encoder**: Attention-based (Transformer-style) — equivalent to fully-connected GAT with masking. No PyTorch Geometric dependency. Heterogeneous node types (agent, lane) via type-specific projections + learned type embeddings.
- **RSSM**: Unchanged from original DITTO (DreamerV2-style categorical stochastic states). Encoder-agnostic — works with any fixed-size embedding input.
- **Continuous actor**: Gaussian policy with `tanh`-bounded actions (steering + acceleration), replacing the discrete categorical actor for Atari.
- **DITTO reward**: `max_cos(z_E, z_π)` in RSSM deterministic state space. Domain-agnostic — works identically for graph vs. pixel latents.
- **Synthetic data**: Built-in highway scenario generator for testing the pipeline without real driving data.

## Installation

```bash
pip install -r requirements.txt
```

## Original DITTO

The original pixel-based DITTO code for Atari is in `src/` (top level). See the [original paper](https://arxiv.org/abs/2302.03086) for details.

