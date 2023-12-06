import random as rand
from math import sqrt
from typing import Optional

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from rl_spotify_user.user_modeling.user_model import UserModel, UserSampler


class SlateGym(gym.Env):
    def __init__(
        self,
        user_sampler,
        doc_sampler,
        num_candidates: int = 10,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.user_sampler = user_sampler
        self.doc_sampler = doc_sampler
        self.device = device
        self.num_candidates = num_candidates

        # initialized by reset
        self.curr_user: UserModel
        self.cdocs_feature: torch.Tensor
        self.cdocs_quality: torch.Tensor
        self.cdocs_length: torch.Tensor

    def step(
        self,
        slate: torch.Tensor,
        cdocs_subset_idx: Optional[torch.Tensor] = None,
    ):
        if cdocs_subset_idx is not None:
            cdocs_feature = self.cdocs_feature[cdocs_subset_idx, :]
            cdocs_quality = self.cdocs_quality[cdocs_subset_idx]
            cdocs_length = self.cdocs_length[cdocs_subset_idx]
        else:
            cdocs_feature = self.cdocs_feature
            cdocs_quality = self.cdocs_quality
            cdocs_length = self.cdocs_length

        cdocs_feature = cdocs_feature[slate, :]
        cdocs_quality = cdocs_quality[slate]
        cdocs_length = cdocs_length[slate]

        diverse_topics = torch.sum(torch.sum(cdocs_feature, dim=0) > 0)
        topic_value = torch.nonzero(torch.sum(cdocs_feature, dim=0) > 0).squeeze()

        # select from the slate on item following the user choice model
        self.curr_user.choice_model.score_documents(
            self.curr_user.get_state(), cdocs_feature
        )
        selected_doc_idx = self.curr_user.choice_model.choose_document()

        if selected_doc_idx == self.curr_user.choice_model.no_selection_token:
            # print("No document selected")
            response = self.curr_user.response_model.generate_null_response()
            self.curr_user.update_budget_noselection()
            # create a fake selected_doc_feature of all zeros
            selected_doc_feature = torch.zeros(cdocs_feature.shape[1])

            selected_doc_quality = 0
        else:
            # print("Document selected")
            # An item has been selected
            # TODO:remove, but can be userful for debugging
            doc_id = slate[selected_doc_idx]

            # checnum_candidates if user has selected a document
            selected_doc_feature = cdocs_feature[selected_doc_idx, :]
            selected_doc_quality = cdocs_quality[selected_doc_idx]
            selected_doc_length = cdocs_length[selected_doc_idx]
            # TODO: remove generate topic response and fix it in the response model
            selected_doc_feature_reshaped = selected_doc_feature.view(1, -1)
            self.slate_items = torch.cat(
                (self.slate_items, selected_doc_feature_reshaped), dim=0
            )
            response = self.curr_user.response_model.generate_response(
                self.curr_user.get_state(),
                selected_doc_feature,
                self.slate_items,
                doc_quality=selected_doc_quality,
            )

            # update user state
            self.curr_user.state_model.update_state(
                selected_doc_feature=selected_doc_feature
            )

            self.curr_user.update_budget(response, int(selected_doc_length.item()))

        is_terminal = self.curr_user.is_terminal()
        info = {}

        return (
            selected_doc_feature,
            selected_doc_quality,
            response,
            is_terminal,
            False,
            info,
            diverse_topics,
            topic_value,
        )

    def reset(self) -> None:
        # 1) sample user
        self.slate_items = torch.zeros(1, 20)
        user = self.user_sampler.sample_user()
        self.curr_user = user
        # 2) reinitialize user budget and user state
        user.budget = user.init_budget()
        self.curr_user.state_model.reset_state()
        # 3) sample candidate documents
        (
            self.cdocs_feature,
            self.cdocs_quality,
            self.cdocs_length,
        ) = self.doc_sampler.sample_documents(self.num_candidates)

    def render(self):
        raise NotImplementedError()

    def get_candidate_docs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.cdocs_feature, self.cdocs_quality, self.cdocs_length
