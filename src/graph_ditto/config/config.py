"""
Config builder for Graph-DITTO.

Reads YAML config files and constructs typed config objects for
the world model, encoder, decoder, agents, and training loops.
"""

from pathlib import Path

import yaml
from ml_collections import ConfigDict


def build_config(config_path: str) -> ConfigDict:
    """Load a YAML config file and build a structured ConfigDict."""
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    mode = raw["mode"]  # "option1" or "option2"
    is_multi_agent = mode == "option2"

    # --- Scene config ---
    scene = ConfigDict(raw["scene"])

    # --- Encoder config ---
    encoder_config = ConfigDict(raw["encoder"])

    # --- RSSM config ---
    rssm_config = ConfigDict(raw["rssm"])

    # --- Decoder config ---
    decoder_config = ConfigDict(raw["decoder"])

    # --- World model config ---
    wm_config = ConfigDict()
    wm_config.multi_agent = is_multi_agent
    wm_config.action_dim = scene.action_dim
    wm_config.kl_weight = raw["loss_weights"]["kl_weight"]
    wm_config.kl_balance = raw["loss_weights"]["kl_balance"]
    wm_config.encoder_config = encoder_config
    wm_config.rssm_config = rssm_config
    wm_config.decoder_config = decoder_config

    if is_multi_agent:
        ma = raw["multi_agent"]
        wm_config.aggregated_action_dim = ma["aggregated_action_dim"]
        wm_config.agent_updater_hidden_dim = ma["agent_updater_hidden_dim"]

    # Features dim = deter_dim + stoch_dim * stoch_rank
    features_dim = rssm_config.deter_dim + rssm_config.stoch_dim * rssm_config.stoch_rank

    # --- Data / dataset config ---
    data_config = ConfigDict(raw["data"])
    data_config.max_agents = scene.max_agents
    data_config.max_lanes = scene.max_lanes
    data_config.agent_feat_dim = scene.agent_feat_dim
    data_config.lane_feat_dim = scene.lane_feat_dim
    data_config.action_dim = scene.action_dim
    data_config.ego_centric = scene.ego_centric
    data_config.seq_length = raw["wm_training"]["seq_length"]

    # --- WM training config ---
    wm_train_config = ConfigDict(raw["wm_training"])
    wm_train_config.train_device = data_config.train_device

    # --- Agent config ---
    agent_raw = raw["agent"]
    agent_config = ConfigDict()
    agent_config.obs_dim = features_dim
    agent_config.action_dim = scene.action_dim
    agent_config.hidden_dim = agent_raw["hidden_dim"]
    agent_config.layers = agent_raw["layers"]
    agent_config.action_scale = agent_raw["action_scale"]
    if is_multi_agent:
        agent_config.agent_embed_dim = agent_raw.get("agent_embed_dim", encoder_config.hidden_dim)
        agent_config.latent_dim = features_dim

    # --- Policy training config ---
    policy_config = ConfigDict(raw["policy_training"])
    policy_config.train_device = data_config.train_device
    policy_config.max_agents = scene.max_agents
    policy_config.action_dim = scene.action_dim

    # --- Top-level config ---
    config = ConfigDict()
    config.mode = mode
    config.multi_agent = is_multi_agent
    config.scene = scene
    config.wm_config = wm_config
    config.data_config = data_config
    config.wm_train_config = wm_train_config
    config.agent_config = agent_config
    config.policy_config = policy_config
    config.features_dim = features_dim
    config.encoder_hidden_dim = encoder_config.hidden_dim

    return config
