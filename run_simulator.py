import random
import hydra
import torch
import numpy as np
import logging
import signal

from datetime import timedelta
from timeit import default_timer as timer
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from utils import utils
from experiment import ExperimentManager, Metric
from models.actor_critic_nn import Critic
from simulators.pid_controller import PIDController
from simulators.positional_pid_controller import PositionalPIDController
from agent_policy_factory import make_actor_and_agent

log = logging.getLogger(__name__)


def apply_actor_lr_multiplier(cfg: DictConfig, logger=None) -> bool:
    actor_lr_multiplier = OmegaConf.select(cfg, "agent.actor_lr_multiplier")
    if actor_lr_multiplier is None:
        return False

    cfg.agent.actor_lr = float(cfg.agent.critic_replay_lr) * float(actor_lr_multiplier)
    if logger is not None:
        logger.info(
            "Using actor_lr_multiplier=%s -> actor_lr=%s (critic_replay_lr=%s)",
            actor_lr_multiplier,
            cfg.agent.actor_lr,
            cfg.agent.critic_replay_lr,
        )
    return True


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    start = timer()
    args = cfg.args
    if HydraConfig.get().mode.value == 2:  # check whether its a sweep
        args.run += HydraConfig.get().job.num
        log.info(f'Running sweep... Run ID: {args.run}')

    apply_actor_lr_multiplier(cfg, logger=log)

    utils.set_seed(args.seed)
    torch.set_num_threads(args.n_threads)

    log.info(OmegaConf.to_yaml({**args, **cfg.agent, **cfg.policy}))
    exp = ExperimentManager(args.db_name, {**args,**cfg.agent, **cfg.policy}, args.db_prefix, cfg.db)
    tables = {}
    for table_name in list(cfg.schema.keys()):
        columns = cfg.schema[table_name].columns
        primary_keys = cfg.schema[table_name].primary_keys
        tables[table_name] = Metric(table_name, columns, primary_keys, exp)

    simulator = PositionalPIDController(reward_noise=args.noise)
    critic = Critic(n_actions=args.n_actions,
                    hidden_units=cfg.agent.critic_hidden_size,
                    hidden_layers=cfg.agent.critic_hidden_layers,
                    lr=cfg.agent.critic_lr,
                    lr_replay=cfg.agent.critic_replay_lr)

    policy, actor, agent = make_actor_and_agent(
        cfg=cfg,
        args=args,
        critic=critic,
    )

    reward_moving_avg = -1
    all_rewards = []
    run_status = "finished"
    step = -1
    action = [np.nan] * args.n_actions
    reward = float("nan")

    def _terminate_handler(signum, _frame):
        nonlocal run_status
        if run_status == "finished":
            run_status = f"killed_signal_{signum}"
        raise SystemExit(128 + signum)

    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _terminate_handler)
    signal.signal(signal.SIGTERM, _terminate_handler)

    try:
        for step in range(0, args.steps):
            action, log_probs = agent.act()
            all_rewards.append(reward := simulator.step(action))
            agent.update_value_baseline(reward)

            if reward_moving_avg == -1:
                reward_moving_avg = reward
            else:
                reward_moving_avg = reward_moving_avg * 0.95 + reward * 0.05

            error, _ = agent.update_critic(action, reward, log_probs)
            if step % 10 == 0:
                tables["errors"].add_data([args.run, step, *action, error, reward, reward_moving_avg])
                tables["policy"].add_data([args.run, step, *policy.get_params()])

            # perform batch gd updates and log them
            if step > cfg.agent.replay_start_step:
                if cfg.agent.n_critic_updates == -1: # variable number of critic updates
                    replay_errs = agent.update_critic_using_buffer(min(cfg.agent.batch_size, len(agent.replay_buffer)))
                else: # do fixed number of critic updates
                    replay_errs = agent.update_critic_fixed_UTD(min(cfg.agent.batch_size, len(agent.replay_buffer)), cfg.agent.n_critic_updates)
                if step % 10 == 0:
                    tables["replay_errors"].add_data([args.run, step, len(replay_errs)-1,
                                                       replay_errs[1], replay_errs[-1]])

            agent.update_actor()

            if step > cfg.agent.replay_start_step and step % 100 == 0:
                log.info(f'[step: {step}] [error: {error:.4f}] [avg_reward: {reward_moving_avg:.4f}] ' \
                         f'[replay_steps: {len(replay_errs)-1}] [replay_error: {np.mean(replay_errs):.4f}]'\
                         f'[action (p,i,d): {tuple(action)}] [reward: {reward:.4f}]')

            # send everything we have accumulated so far to the db
            if step % 1000 == 0:
                tables["errors"].commit_to_database()
                tables["policy"].commit_to_database()
                tables["replay_errors"].commit_to_database()

    except BaseException as exc:
        if run_status == "finished":
            run_status = f"crashed_{type(exc).__name__}"
        raise
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)

        total_time = timedelta(seconds=timer() - start).seconds / 60
        log.info(f'Total time taken: {total_time}  minutes')
        if all_rewards:
            auc_final_10perc = float(np.mean(all_rewards[-max(1, int(len(all_rewards) * 0.1)):]))
            auc_final_50perc = float(np.mean(all_rewards[-max(1, int(len(all_rewards) * 0.5)):]))
            mean_reward = float(np.mean(all_rewards))
        else:
            auc_final_10perc = float("nan")
            auc_final_50perc = float("nan")
            mean_reward = float("nan")
        tables["summary"].add_data([args.run, step, *action, reward, mean_reward, auc_final_10perc, auc_final_50perc, run_status, total_time])
        tables["errors"].commit_to_database()
        tables["policy"].commit_to_database()
        tables["replay_errors"].commit_to_database()
        tables["summary"].commit_to_database()


if __name__ == "__main__":
    main()
