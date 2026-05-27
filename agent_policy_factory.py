import copy

from models.actor_critic_nn import Actor
from models.base_models.replay_buffer import ReplayBuffer


def make_policy(cfg, args):
    actor_policy = cfg.policy.actor_policy
    actor_policy_mean_init = cfg.policy.actor_policy_mean_init
    actor_policy_shape_init = cfg.policy.actor_policy_shape_init
    log_prob_reduction = cfg.agent.log_prob_reduction
    action_min = [args.action_min] * args.n_actions
    action_max = [args.action_max] * args.n_actions

    match actor_policy:
        case "gaussian":
            from models.base_models.gaussian import Gaussian
            return Gaussian(
                num_actions=args.n_actions,
                mean_init=actor_policy_mean_init,
                std_init=actor_policy_shape_init,
                action_min=action_min,
                action_max=action_max,
                log_prob_reduction=log_prob_reduction,
            )
        case "gaussian_squashed":
            from models.base_models.squashed_gaussian import SquashedGaussian
            return SquashedGaussian(
                num_actions=args.n_actions,
                mean_init=actor_policy_mean_init,
                std_init=actor_policy_shape_init,
                action_min=action_min,
                action_max=action_max,
                log_prob_reduction=log_prob_reduction,
            )
        case "gaussian_fixed":
            from models.base_models.gaussian import GaussianFixedStd
            return GaussianFixedStd(
                num_actions=args.n_actions,
                mean_init=actor_policy_mean_init,
                std_init=actor_policy_shape_init,
                anneal_coef=cfg.agent.anneal_coef,
                action_min=action_min,
                action_max=action_max,
            )
        case "student":
            from models.base_models.student import Student
            return Student(
                num_actions=args.n_actions,
                mean_init=actor_policy_mean_init,
                shape_init=actor_policy_shape_init,
                df_init=cfg.policy.actor_policy_df_init,
                action_min=action_min,
                action_max=action_max,
                fixed_df=cfg.policy.actor_policy_fixed_df,
                log_prob_reduction=log_prob_reduction,
            )
        case "student_squashed":
            from models.base_models.squashed_student import SquashedStudent
            return SquashedStudent(
                num_actions=args.n_actions,
                mean_init=actor_policy_mean_init,
                shape_init=actor_policy_shape_init,
                df_init=cfg.policy.actor_policy_df_init,
                action_min=action_min,
                action_max=action_max,
                fixed_df=cfg.policy.actor_policy_fixed_df,
                log_prob_reduction=log_prob_reduction,
            )
        case "beta":
            from models.base_models.beta import BetaNN
            return BetaNN(
                num_actions=args.n_actions,
                mean=actor_policy_mean_init,
                shape=actor_policy_shape_init,
                action_min=[args.action_min],
                action_max=[args.action_max],
                log_prob_reduction=log_prob_reduction,
            )
        case "softmax":
            from models.base_models.softmax import SoftmaxPolicyWrapper
            return SoftmaxPolicyWrapper(
                num_actions=args.n_actions,
                action_min=action_min,
                action_max=action_max,
            )
        case "softmax_kl":
            from models.base_models.softmax import SoftmaxPolicyWrapper
            return SoftmaxPolicyWrapper(num_actions=args.n_actions, kl=True)
        case _:
            raise NotImplementedError(f"Unknown policy: {actor_policy}")


def make_agent(cfg, critic, actor):
    replay_buffer = ReplayBuffer()

    match cfg.agent.name:
        case "greedyac":
            from agents.greedyac_agent import GreedyACAgent
            return GreedyACAgent(
                args=cfg.agent,
                critic=critic,
                proposal_actor=actor,
                behavior_actor=copy.deepcopy(actor),
                replay_buffer=replay_buffer,
                use_true_reward=cfg.agent.use_true_reward,
            )
        case "greedyac_md":
            from agents.greedyac_agent import GreedyACMirrorDescentAgent
            kl_penalty_coef = cfg.agent.kl_penalty_coef
            if cfg.agent.use_eta:
                kl_penalty_coef *= cfg.agent.critic_replay_lr
            return GreedyACMirrorDescentAgent(
                args=cfg.agent,
                critic=critic,
                proposal_actor=actor,
                behavior_actor=copy.deepcopy(actor),
                replay_buffer=replay_buffer,
                kl_penalty_coef=kl_penalty_coef,
                md_period=cfg.agent.mirror_descent_period,
            )
        case "sac":
            from agents.sac_agent import SACAgent
            return SACAgent(
                args=cfg.agent,
                critic=critic,
                actor=actor,
                replay_buffer=replay_buffer,
                entropy_coef=cfg.agent.entropy_coef,
                use_reparameterisation_trick=cfg.policy.reparameterisation_trick,
                use_true_reward=False,
            )
        case "sac_md":
            from agents.sac_agent import SACMirrorDescentAgent
            return SACMirrorDescentAgent(
                args=cfg.agent,
                critic=critic,
                actor=actor,
                replay_buffer=replay_buffer,
                entropy_coef=cfg.agent.entropy_coef,
                use_reparameterisation_trick=cfg.policy.reparameterisation_trick,
                kl_penalty_coef=cfg.agent.kl_penalty_coef,
                md_period=cfg.agent.mirror_descent_period,
            )
        case "mpo":
            from agents.mpo_agent import MPOAgent
            return MPOAgent(
                args=cfg.agent,
                critic=critic,
                actor=actor,
                replay_buffer=replay_buffer,
                kl_coef=cfg.agent.kl_coef,
                use_true_reward=cfg.agent.use_true_reward,
            )
        case "adaptive_mpo":
            from agents.mpo_adaptive_agent import MPOAdaptiveAgent
            return MPOAdaptiveAgent(
                args=cfg.agent,
                critic=critic,
                actor=actor,
                replay_buffer=replay_buffer,
                kl_coef=cfg.agent.kl_coef,
                use_true_reward=cfg.agent.use_true_reward,
            )            
        case "vmpo":
            from agents.vmpo_agent import VMPOAgent
            return VMPOAgent(
                args=cfg.agent,
                critic=critic,
                actor=actor,
                replay_buffer=replay_buffer,
            )
        case "ddpg":
            from agents.ddpg_agent import DDPGAgent
            return DDPGAgent(
                args=cfg.agent,
                critic=critic,
                actor=actor,
                replay_buffer=replay_buffer,
            )
        case "ddpg_true_reward":
            try:
                from agents.ddpg_true_reward_agent import DDPGTrueRewardAgent
            except ModuleNotFoundError as exc:
                raise NotImplementedError(
                    "Agent 'ddpg_true_reward' is configured but "
                    "agents/ddpg_true_reward_agent.py is missing."
                ) from exc
            return DDPGTrueRewardAgent(
                args=cfg.agent,
                critic=critic,
                actor=actor,
                replay_buffer=replay_buffer,
            )
        case "ppo":
            from agents.ppo_agent import PPOAgent
            return PPOAgent(
                args=cfg.agent,
                critic=critic,
                actor=actor,
                replay_buffer=replay_buffer,
                clip_coef=cfg.agent.clip_coef,
                entropy_coef=cfg.agent.entropy_coef,
                v_coef=cfg.agent.v_coef,
                onpolicy_samples=cfg.agent.onpolicy_samples,
            )
        case "ppo_md":
            from agents.ppo_agent import PPOMirrorDescentAgent
            return PPOMirrorDescentAgent(
                args=cfg.agent,
                critic=critic,
                actor=actor,
                replay_buffer=replay_buffer,
                clip_coef=cfg.agent.clip_coef,
                entropy_coef=cfg.agent.entropy_coef,
                v_coef=cfg.agent.v_coef,
                onpolicy_samples=cfg.agent.onpolicy_samples,
                kl_penalty_coef=cfg.agent.kl_penalty_coef,
                md_period=cfg.agent.mirror_descent_period,
            )
        case "reinforce":
            from agents.reinforce_agent import ReinforceAgent
            return ReinforceAgent(
                args=cfg.agent,
                critic=critic,
                actor=actor,
                replay_buffer=replay_buffer,
            )
        case "aclambda":
            from agents.aclambda_agent import ACLambdaAgent
            return ACLambdaAgent(
                args=cfg.agent,
                critic=critic,
                actor=actor,
                replay_buffer=replay_buffer,
                lamda=cfg.agent.lamda,
            )
        case _:
            raise NotImplementedError(f"Unknown agent: {cfg.agent.name}")


def make_actor_and_agent(cfg, args, critic):
    policy = make_policy(cfg=cfg, args=args)
    actor = Actor(policy=policy, lr=cfg.agent.actor_lr)
    agent = make_agent(cfg=cfg, critic=critic, actor=actor)
    return policy, actor, agent
