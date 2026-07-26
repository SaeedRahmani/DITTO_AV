"""DITTO-AV: Offline imitation learning with world models for autonomous driving.

Pipeline: collect expert driving data -> train a vector RSSM world model ->
learn a policy fully offline by latent-matching imitation inside the world
model (single-mode DITTO reward, or the multimodal nearest-mode reward) ->
evaluate closed-loop in the real environment.
"""

__version__ = "0.1.0"
