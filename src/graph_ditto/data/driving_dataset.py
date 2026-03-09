"""
Driving dataset for Graph-DITTO.

Loads driving episodes (either synthetic or from real datasets),
constructs scene graphs, and provides sequences for RSSM training
and DITTO policy learning.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from graph_ditto.data.scene_graph import SceneGraphBuilder
from graph_ditto.data.synthetic import SyntheticDrivingGenerator


class DrivingDataset(Dataset):
    """
    Dataset of driving scenes for world model and policy training.

    Stores pre-processed, padded scene graphs as contiguous tensors
    for efficient batching with the RSSM.
    """

    def __init__(self, conf):
        self.seq_length = conf.seq_length
        self.max_agents = conf.max_agents
        self.max_lanes = conf.max_lanes
        self.agent_feat_dim = conf.agent_feat_dim
        self.lane_feat_dim = conf.lane_feat_dim
        self.action_dim = conf.action_dim
        self.ego_centric = getattr(conf, "ego_centric", True)

        self.graph_builder = SceneGraphBuilder(
            max_agents=self.max_agents,
            max_lanes=self.max_lanes,
            agent_feat_dim=self.agent_feat_dim,
            lane_feat_dim=self.lane_feat_dim,
        )

        # Build data from episodes
        self.data = self._build_data(conf)
        self.num_transitions = len(self.data["agent_features"])
        print(f"DrivingDataset: {self.num_transitions} transitions loaded")

    def _build_data(self, conf):
        """Load or generate episodes and convert to padded tensors."""
        data_source = getattr(conf, "data_source", "synthetic")

        if data_source == "synthetic":
            generator = SyntheticDrivingGenerator(
                n_episodes=getattr(conf, "n_episodes", 100),
                episode_length=getattr(conf, "episode_length", 100),
                seed=getattr(conf, "seed", 42),
            )
            episodes = generator.generate_dataset()
        else:
            raise NotImplementedError(
                f"Data source '{data_source}' not yet supported. "
                "Implement a loader for your driving dataset format."
            )

        return self._process_episodes(episodes)

    def _process_episodes(self, episodes):
        """Convert raw episodes into padded tensor data."""
        all_agent_features = []
        all_agent_masks = []
        all_lane_features = []
        all_lane_masks = []
        all_ego_masks = []
        all_actions = []
        all_resets = []

        for ep in episodes:
            agent_states = ep["agent_states"]   # (T, N, 8)
            lane_points = ep["lane_points"]     # (N_l, 6)
            ego_actions = ep["ego_actions"]     # (T, 2)
            resets = ep["resets"]               # (T,)
            T = agent_states.shape[0]

            for t in range(T):
                agents_t = agent_states[t]
                lanes_t = lane_points

                if self.ego_centric:
                    agents_t, lanes_t = self.graph_builder.normalize_to_ego(
                        agents_t, lanes_t, ego_idx=0
                    )

                graph = self.graph_builder.build_graph(agents_t, lanes_t, ego_idx=0)

                all_agent_features.append(graph["agent_features"])
                all_agent_masks.append(graph["agent_mask"])
                all_lane_features.append(graph["lane_features"])
                all_lane_masks.append(graph["lane_mask"])
                all_ego_masks.append(graph["ego_mask"])
                all_actions.append(ego_actions[t])
                all_resets.append(resets[t])

        data = {
            "agent_features": torch.tensor(np.array(all_agent_features)),
            "agent_mask": torch.tensor(np.array(all_agent_masks)),
            "lane_features": torch.tensor(np.array(all_lane_features)),
            "lane_mask": torch.tensor(np.array(all_lane_masks)),
            "ego_mask": torch.tensor(np.array(all_ego_masks)),
            "action": torch.tensor(np.array(all_actions, dtype=np.float32)),
            "reset": torch.tensor(np.array(all_resets)),
        }

        return data

    def __len__(self):
        return self.num_transitions

    def __getitem__(self, idx):
        indices = [(idx + i) % self.num_transitions for i in range(self.seq_length)]
        return {
            "agent_features": self.data["agent_features"][indices],
            "agent_mask": self.data["agent_mask"][indices],
            "lane_features": self.data["lane_features"][indices],
            "lane_mask": self.data["lane_mask"][indices],
            "ego_mask": self.data["ego_mask"][indices],
            "action": self.data["action"][indices],
            "reset": self.data["reset"][indices],
        }


class DrivingFeaturizer:
    """
    Extracts RSSM latent features from a trained world model for a driving dataset.

    Analogous to the Featurizer in original DITTO: runs the world model encoder
    over the dataset to produce latent features for policy training.
    """

    def __init__(self, world_model, device="cuda"):
        self.wm = world_model
        self.device = device

    @torch.inference_mode()
    def featurize(self, dataset, batch_size=32):
        """Run the world model encoder over the dataset to get latent features.

        Returns:
            dict of torch tensors:
                features:     (N, features_dim)
                agent_embeds: (N, max_agents, agent_embed_dim)
                agent_masks:  (N, max_agents)
                actions:      (N, action_dim)
                resets:       (N,)
        """
        from graph_ditto.data.common import SequentialSampler
        from torch.utils.data import DataLoader

        saved_seq_length = dataset.seq_length
        dataset.seq_length = 1

        sampler = SequentialSampler(
            len(dataset), seq_length=1, batch_size=batch_size, init_idx=0
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=driving_collate,
        )

        N = len(dataset)
        features_dim = self.wm.features_dim
        agent_embed_dim = self.wm.encoder.hidden_dim
        max_agents = dataset.max_agents
        action_dim = dataset.action_dim

        all_features = torch.zeros(N, features_dim)
        all_agent_embeds = torch.zeros(N, max_agents, agent_embed_dim)
        all_agent_masks = torch.zeros(N, max_agents, dtype=torch.bool)
        all_actions = torch.zeros(N, action_dim)
        all_resets = torch.zeros(N, dtype=torch.bool)

        in_states = self.wm.init_state(batch_size)
        idx = 0

        for batch in loader:
            obs = {k: v.to(self.device) for k, v in batch.items()}
            features, out_states, agent_embeds = self.wm(obs, in_states)
            in_states = tuple(s.detach() for s in out_states)

            # features: (1, B, feat), agent_embeds: (1, B, N_a, E)
            feat = features.squeeze(0).cpu()       # (B, feat)
            ae = agent_embeds.squeeze(0).cpu()      # (B, N_a, E)
            masks = obs["agent_mask"].squeeze(0).cpu()  # (B, N_a)
            acts = obs["action"].squeeze(0).cpu()   # (B, act)
            rsts = obs["reset"].squeeze(0).cpu()    # (B,)

            bs = feat.shape[0]
            all_features[idx : idx + bs] = feat
            all_agent_embeds[idx : idx + bs] = ae
            all_agent_masks[idx : idx + bs] = masks
            all_actions[idx : idx + bs] = acts
            all_resets[idx : idx + bs] = rsts
            idx += bs

        dataset.seq_length = saved_seq_length

        return {
            "features": all_features,
            "agent_embeds": all_agent_embeds,
            "agent_masks": all_agent_masks,
            "actions": all_actions,
            "resets": all_resets,
        }


def driving_collate(batch):
    """Collate function that transposes batch and time dims: (B, T, ...) -> (T, B, ...)"""
    from torch.utils.data._utils.collate import default_collate

    collated = default_collate(batch)
    return {k: torch.transpose(v, 0, 1) for k, v in collated.items()}
