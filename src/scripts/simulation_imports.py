import argparse
import configparser
import os
import pickle
import random
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
import yaml
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from rl_spotify_user.agent_modeling.dqn_agent import (
    DQNAgent,
    ReplayMemoryDataset,
    Transition,
)
from rl_spotify_user.agent_modeling.slate_generator import (  # DiverseSlateGenerator,; GreedySlateGenerator,; OptimalSlateGenerator,
    RandomSlateGenerator,
    TopKSlateGenerator,
    GreedySlateGenerator,
)
from rl_spotify_user.agent_modeling.wp_agent import WolpertingerActor, ActorAgent
from rl_spotify_user.agent_modeling.wp_slate_agent import (
    WolpertingerActorSlate,
    ActorAgentSlate,
)
from rl_spotify_user.document_modeling.document_sampler import DocumentSampler
from rl_spotify_user.simulation_environment.environment import SlateGym
from rl_spotify_user.user_modeling.choice_model import (
    CosineSimilarityChoiceModel,
    DotProductChoiceModel,
)
from rl_spotify_user.user_modeling.features_gen import (
    NormalUserFeaturesGenerator,
    UniformFeaturesGenerator,
    BinaryFeaturesGenerator,
)
from rl_spotify_user.user_modeling.response_model import (
    CosineResponseModel,
    DotProductResponseModel,
    WeightedDotProductResponseModel,
)
from rl_spotify_user.user_modeling.user_model import UserSampler
from rl_spotify_user.user_modeling.user_state import (
    ObservableUserState,
    BoredomObservableUserState,
)
from rl_spotify_user.utils import save_run, save_run_wa

class_name_to_class = {
    "ObservableUserState": ObservableUserState,
    "BoredomObservableUserState": BoredomObservableUserState,
    "DotProductChoiceModel": DotProductChoiceModel,
    "DotProductResponseModel": DotProductResponseModel,
    "TopKSlateGenerator": TopKSlateGenerator,
    "GreedySlateGenerator": GreedySlateGenerator,
    "RandomSlateGenerator": RandomSlateGenerator,
    "NormalFeaturesGenerator": NormalUserFeaturesGenerator,
    "UniformFeaturesGenerator": UniformFeaturesGenerator,
    "BinaryFeaturesGenerator": BinaryFeaturesGenerator,
    # "DiverseSlateGenerator": DiverseSlateGenerator,
    # "GreedySlateGenerator": GreedySlateGenerator,
    # "OptimalSlateGenerator": OptimalSlateGenerator,
    "WeightedDotProductResponseModel": WeightedDotProductResponseModel,
}
load_dotenv()
