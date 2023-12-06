from rl_recsys.user_modeling.features_gen import UniformFeaturesGenerator
from scripts.simulation_imports import *


def optimize_model(batch, batch_size):
    optimizer.zero_grad()

    (
        state_batch,  # [batch_size, num_item_features]
        selected_doc_feat_batch,  # [batch_size, num_item_features]
        candidates_batch,  # [batch_size, num_candidates, num_item_features]
        satisfaction_batch,  # [batch_size, 1]
        next_state_batch,  # [batch_size, num_item_features]
    ) = batch

    # Q(s, a): [batch_size, 1]
    q_val = agent.compute_q_values(
        state_batch, selected_doc_feat_batch, use_policy_net=True
    )  # type: ignore

    # with torch.no_grad():
    cand_qtgt_list = []
    for b in range(next_state_batch.shape[0]):
        next_state = next_state_batch[b, :]
        candidates = candidates_batch[b, :, :]
        # candidates, _ = actor.k_nearest(
        #     next_state, candidates, use_actor_policy_net=False
        # )

        candidates = candidates[:, :NUM_ITEM_FEATURES]

        next_state_rep = next_state.repeat((candidates.shape[0], 1))
        cand_qtgt = agent.compute_q_values(
            next_state_rep, candidates, use_policy_net=False
        )  # type: ignore
        # cand_qtgt = agent.compute_q_values(
        #     next_state_rep,
        #     actor.compute_proto_action(next_state_rep, use_actor_policy_net=False),
        # )

        choice_model.score_documents(next_state, candidates)
        # [num_candidates, 1]
        scores_tens = torch.Tensor(choice_model.scores).to(DEVICE).unsqueeze(dim=1)
        # max over Q(s', a)
        scores_tens = torch.softmax(scores_tens, dim=0)

        # cand_qtgt_list.append((cand_qtgt * scores_tens).max())

        topk = torch.topk((cand_qtgt * scores_tens), dim=0, k=SLATE_SIZE)
        curr_q_tgt = topk.values
        topk_idx = topk.indices
        p_sum = scores_tens[topk_idx, :].squeeze().sum()
        # normalize curr_q_tgt to sum to 1
        curr_q_tgt = torch.sum(curr_q_tgt / p_sum)

        # curr_q_tgt = torch.topk(
        #     (cand_qtgt * scores_tens), dim=0, k=SLATE_SIZE
        # ).values.sum()

        cand_qtgt_list.append(curr_q_tgt)

    q_tgt = torch.stack(cand_qtgt_list).unsqueeze(dim=1)

    expected_q_values = q_tgt * GAMMA + satisfaction_batch.unsqueeze(dim=1)
    # print("q_val", q_val.mean(dim=0))
    # print("q_tgt", q_tgt.mean(dim=0))
    # print("reward", satisfaction_batch.mean())

    # expected_q_values = q_tgt
    loss = criterion(q_val, expected_q_values)

    # Optimize the model
    loss.backward()
    optimizer.step()
    item = actor.compute_proto_slate(state_batch, use_actor_policy_net=True)
    proto_action_tensor = item.reshape(batch_size, 5, 20)
    actor_loss_list = []

    # Taking the average along the third axis to reduce the tensor size
    # proto_action_tensor_2 = torch.mean(proto_action_tensor, axis=1)
    for i in range(batch_size):
        proto_action_tensor_rep = proto_action_tensor[i]
        state = state_batch[i]
        user_state_rep = state.repeat((proto_action_tensor_rep.shape[0], 1))
        q_loss = agent.compute_q_values(
            user_state_rep,
            proto_action_tensor_rep,
            use_policy_net=True,
        )
        with torch.no_grad():
            detached_scores = choice_model.score_documents(
                state, proto_action_tensor_rep
            ).detach()
            # [num_candidates, 1]
        scores_tens_loss = torch.Tensor(detached_scores).to(DEVICE).unsqueeze(dim=1)
        # max over Q(s', a)
        scores_tens_loss = torch.softmax(scores_tens_loss, dim=0)
        v_sum = scores_tens_loss.squeeze().sum()
        actor_loss_list.append(-torch.sum((q_loss * scores_tens_loss) / v_sum))
    actor_loss = torch.tensor(actor_loss_list, requires_grad=True).unsqueeze(1).mean()
    # actor_item_loss = torch.empty(128, 5)
    # for i in range(5):
    #     q_values = -agent.compute_q_values(
    #         state_batch,
    #         proto_action_tensor[:, i, :],
    #         use_policy_net=True,
    # print(q_loss.grad_fn)
    # print(scores_tens_loss.grad_fn)
    # #     )
    # a = 10
    #     if q_values.shape != (128, 1):
    #         raise ValueError(f"Shape mismatch in q_values for column {i}")

    #     # Assign q_values to the corresponding column
    #     actor_item_loss[:, i : i + 1] = -q_values
    # actor_loss = torch.mean(actor_item_loss, dim=1, keepdim=True).mean()
    actor_loss.backward()
    actor_optimizer.step()
    return loss, actor_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config_path = "src/scripts/config/config.yaml"
    parser.add_argument(
        "--config",
        type=str,
        default=config_path,
        help="Path to the config file.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    parameters = config["parameters"]
    SEEDS = parameters["seeds"]
    for seed in SEEDS:
        pl.seed_everything(seed)
        ######## User related parameters ########
        state_model_cls = parameters["state_model_cls"]
        choice_model_cls = parameters["choice_model_cls"]
        response_model_cls = parameters["response_model_cls"]
        resp_amp_factor = parameters["resp_amp_factor"]
        user_feature_cls = parameters["user_feature_model_cls"]

        ######## Environment related parameters ########
        SLATE_SIZE = parameters["slate_size"]
        NUM_CANDIDATES = parameters["num_candidates"]
        NUM_USERS = parameters["num_users"]
        NUM_ITEM_FEATURES = parameters["num_item_features"]
        SESS_BUDGET = parameters["sess_budget"]
        NUM_USER_FEATURES = parameters["num_user_features"]
        ALPHA_RESPONSE = parameters["alpha_response"]
        NEAREST_NEIGHBOURS = parameters["nearest_neighbours"]
        GROUPING = parameters["grouping"]

        ######## Training related parameters ########
        REPLAY_MEMORY_CAPACITY = parameters["replay_memory_capacity"]
        BATCH_SIZE = parameters["batch_size"]
        GAMMA = parameters["gamma"]
        TAU = parameters["tau"]
        LR = float(parameters["lr"])
        NUM_EPISODES = parameters["num_episodes"]
        WARMUP_BATCHES = parameters["warmup_batches"]
        DEVICE = parameters["device"]
        DEVICE = torch.device(DEVICE)
        print("DEVICE: ", DEVICE)
        ######## Models related parameters ########
        slate_gen_model_cls = parameters["slate_gen_model_cls"]

        ######## Init_wandb ########
        RUN_NAME = f"Topic_GAMMA_{GAMMA}_{seed}_ALPHA_{ALPHA_RESPONSE}_{GROUPING}"
        wandb.init(project="rl_recsys", config=config["parameters"], name=RUN_NAME)

        ################################################################
        user_feat_gen = BinaryFeaturesGenerator()
        state_model_cls = class_name_to_class[state_model_cls]
        choice_model_cls = class_name_to_class[choice_model_cls]
        response_model_cls = class_name_to_class[response_model_cls]

        state_model_kwgs = {}
        choice_model_kwgs = {}
        response_model_kwgs = {"amp_factor": resp_amp_factor, "alpha": ALPHA_RESPONSE}

        user_sampler = UserSampler(
            user_feat_gen,
            state_model_cls,
            choice_model_cls,
            response_model_cls,
            state_model_kwargs=state_model_kwgs,
            choice_model_kwargs=choice_model_kwgs,
            response_model_kwargs=response_model_kwgs,
            sess_budget=SESS_BUDGET,
            num_user_features=NUM_USER_FEATURES,
        )
        user_sampler.generate_users(num_users=NUM_USERS)

        # TODO: dont really now why needed there we shold use the one associated to the user sampled for the episode
        choice_model = choice_model_cls()
        doc_sampler = DocumentSampler(seed=seed)
        env = SlateGym(
            user_sampler=user_sampler,
            doc_sampler=doc_sampler,
            num_candidates=NUM_CANDIDATES,
            device=DEVICE,
        )

        slate_gen_model_cls = class_name_to_class[slate_gen_model_cls]
        slate_gen = slate_gen_model_cls(slate_size=SLATE_SIZE)

        # input features are 2 * NUM_ITEM_FEATURES since we concatenate the state and one item
        agent = DQNAgent(
            slate_gen=slate_gen,
            input_size=2 * NUM_ITEM_FEATURES,
            output_size=1,
            tau=TAU,
        ).to(DEVICE)

        transition_cls = Transition
        replay_memory_dataset = ReplayMemoryDataset(
            capacity=REPLAY_MEMORY_CAPACITY, transition_cls=transition_cls
        )
        replay_memory_dataloader = DataLoader(
            replay_memory_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=replay_memory_dataset.collate_fn,
            shuffle=False,
        )
        actor = ActorAgentSlate(
            nn_dim=[
                20,
                40,
                60,
                80,
                100,
            ],  # observable= [20, 40, 60, 80, 100], weight_decay=1e-4
            k=int(NEAREST_NEIGHBOURS / SLATE_SIZE),
            slate_size=SLATE_SIZE,
        )

        criterion = torch.nn.SmoothL1Loss()
        optimizer = optim.Adam(agent.parameters(), lr=LR)
        actor_optimizer = optim.Adam(actor.parameters(), lr=LR)

        ############################## TRAINING ###################################
        save_dict = defaultdict(list)
        is_terminal = False
        for i_episode in tqdm(range(NUM_EPISODES)):
            (
                satisfaction,
                loss,
                diff_to_best,
                quality,
                actor_loss,
                time_unit_consumed,
            ) = ([], [], [], [], [], [])

            env.reset()
            is_terminal = False
            cum_satisfaction = 0

            cdocs_features, cdocs_quality, cdocs_length = env.get_candidate_docs()
            user_state = torch.Tensor(env.curr_user.get_state()).to(DEVICE)

            max_sess, avg_sess = [], []
            while not is_terminal:
                # print("user_state: ", user_state)
                with torch.no_grad():
                    ########################################
                    satisfactions_candidates = (
                        (1 - ALPHA_RESPONSE)
                        * torch.mm(
                            user_state.unsqueeze(0),
                            cdocs_features.t(),
                        )
                        + ALPHA_RESPONSE * cdocs_quality
                    ).squeeze(0) * resp_amp_factor

                    max_rew = satisfactions_candidates.max()
                    min_rew = satisfactions_candidates.min()
                    mean_rew = satisfactions_candidates.mean()
                    std_rew = satisfactions_candidates.std()

                    max_sess.append(max_rew)
                    avg_sess.append(mean_rew)
                    ########################################
                    cdocs_features_act, candidates = actor.k_nearest(
                        user_state,
                        cdocs_features,
                        slate_size=SLATE_SIZE,
                        use_actor_policy_net=True,
                    )

                    user_state_rep = user_state.repeat((cdocs_features_act.shape[0], 1))

                    q_val = agent.compute_q_values(
                        state=user_state_rep,
                        candidate_docs_repr=cdocs_features_act,
                        use_policy_net=True,
                    )  # type: ignore

                    choice_model.score_documents(
                        user_state=user_state, docs_repr=cdocs_features_act
                    )
                    scores = torch.Tensor(choice_model.scores).to(DEVICE)
                    scores = torch.softmax(scores, dim=0)

                    q_val = q_val.squeeze()
                    slate = agent.get_action(scores, q_val)
                    Q_values = q_val[slate]

                    Q_value_max = Q_values.max()
                    Q_value_min = Q_values.min()
                    Q_value_diff = Q_value_max - Q_value_min
                    # print("slate: ", slate)

                    (
                        selected_doc_feature,
                        doc_quality,
                        response,
                        is_terminal,
                        _,
                        _,
                        diverse_topics,
                        topic_position,
                    ) = env.step(slate, cdocs_subset_idx=candidates)
                    # normalize satisfaction between 0 and 1
                    # response = (response - min_rew) / (max_rew - min_rew)
                    satisfaction.append(response)
                    quality.append(doc_quality)

                    next_user_state = env.curr_user.get_state()
                    # push memory
                    if not torch.all(selected_doc_feature == 0):
                        replay_memory_dataset.push(
                            transition_cls(
                                user_state,  # type: ignore
                                selected_doc_feature,
                                cdocs_features,
                                response,
                                next_user_state,  # type: ignore
                            )
                        )
                    user_state = next_user_state
                    if torch.all(selected_doc_feature == 0):
                        time_unit_consumed.append(-0.5)
                    else:
                        time_unit_consumed.append(4.0)

            # optimize model
            if len(replay_memory_dataset.memory) >= WARMUP_BATCHES * BATCH_SIZE:
                batch = next(iter(replay_memory_dataloader))
                for elem in batch:
                    elem.to(DEVICE)
                batch_loss, batch_actor_loss = optimize_model(batch, BATCH_SIZE)
                agent.soft_update_target_network()
                # actor.soft_update_target_network()
                loss.append(batch_loss)
                actor_loss.append(batch_actor_loss)

            loss = torch.mean(torch.tensor(loss))
            actor_loss = torch.mean(torch.tensor(actor_loss))
            sess_length = np.sum(time_unit_consumed)
            ep_quality = torch.mean(torch.tensor(quality))
            ep_avg_satisfaction = torch.mean(torch.tensor(satisfaction))
            ep_cum_satisfaction = torch.sum(torch.tensor(satisfaction))
            ep_max_avg = torch.mean(torch.tensor(max_sess))
            ep_max_cum = torch.sum(torch.tensor(max_sess))
            ep_avg_avg = torch.mean(torch.tensor(avg_sess))
            ep_avg_cum = torch.sum(torch.tensor(avg_sess))
            cum_normalized = (
                ep_cum_satisfaction / ep_max_cum
                if ep_max_cum > 0
                else ep_max_cum / ep_cum_satisfaction
            )

            log_str = (
                f"Loss: {loss}\n"
                f"Actor_Loss: {actor_loss}\n"
                f"Avg_satisfaction: {ep_avg_satisfaction} - Cum_Rew: {ep_cum_satisfaction}\n"
                f"Max_Avg_satisfaction: {ep_max_avg} - Max_Cum_Rew: {ep_max_cum}\n"
                f"Avg_Avg_satisfaction: {ep_avg_avg} - Avg_Cum_Rew: {ep_avg_cum}\n"
                f"Cumulative_Normalized: {cum_normalized}"
                f"Diverse_topics: {diverse_topics}\n"
            )
            # print(log_str)
            ###########################################################################
            log_dict = {
                "Q_value_max": Q_value_max,
                "Q_value_min": Q_value_min,
                "Q_value_diff": Q_value_diff,
                "quality": ep_quality,
                "avg_satisfaction": ep_avg_satisfaction,
                "cum_satisfaction": ep_cum_satisfaction,
                "max_avg": ep_max_avg,
                "max_cum": ep_max_cum,
                "avg_avg": ep_avg_avg,
                "avg_cum": ep_avg_cum,
                "best_rl_avg_diff": ep_max_avg - ep_avg_satisfaction,
                "best_avg_avg_diff": ep_max_avg - ep_avg_avg,
                "cum_normalized": cum_normalized,
                "session_length": sess_length,
                "diverse_topics": diverse_topics,
            }
            if len(replay_memory_dataset.memory) >= (WARMUP_BATCHES * BATCH_SIZE):
                log_dict["loss"] = loss
                log_dict["actor_loss"] = actor_loss
            wandb.log(log_dict, step=i_episode)

            ###########################################################################
            save_dict["session_length"].append(sess_length)
            save_dict["ep_cum_satisfaction"].append(ep_cum_satisfaction)
            save_dict["ep_avg_satisfaction"].append(ep_avg_satisfaction)
            save_dict["loss"].append(loss)
            save_dict["actor_loss"].append(actor_loss)
            save_dict["best_rl_avg_diff"].append(ep_max_avg - ep_avg_satisfaction)
            save_dict["best_avg_avg_diff"].append(ep_max_avg - ep_avg_avg)
            save_dict["cum_normalized"].append(cum_normalized)

        wandb.finish()
        # directory = f"proto_slate_q_avg_300_20_{ALPHA_RESPONSE}"
        # save_run_wa(
        #     seed=seed,
        #     save_dict=save_dict,
        #     agent=agent,
        #     directory=directory,
        #     actor=actor,
        # )
