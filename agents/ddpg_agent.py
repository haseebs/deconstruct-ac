import torch
import numpy as np
import sys
sys.path.insert(0, "../")
from utils.utils import kl_selection

# i think we can combine some of these AC agents later
class DDPGAgent:
    def __init__(self, args, critic, actor, replay_buffer, kl_penalty_coef=None) -> None:
        self.args = args
        self.n_actor_updates = args.n_actor_updates

        self.max_replay_steps = args.max_replay_steps
        self.err_tolerance_tau = args.err_tolerance_tau

        self.actor = actor
        self.critic = critic
        self.replay_buffer = replay_buffer
        self.kl_penalty_coef = kl_penalty_coef


    @torch.no_grad()
    def act(self) -> list[float]:
        """every time an action has been sampled we anneal the noise"""
        action, log_probs, _, _ = self.actor.policy.sample(num_samples=1)
        self.actor.policy.anneal()
        action = torch.clamp(action, min=torch.FloatTensor(self.actor.policy.action_min), max=torch.FloatTensor(self.actor.policy.action_max))
        return action.tolist(), log_probs.tolist()


    def update_value_baseline(self, reward: float) -> None:
        return

    def update_critic(self, action: list[float], target: float, log_probs: list[float]) -> float:
        self.replay_buffer.append(action, target, log_probs)
        prediction = self.critic.get_qvalues([action])
        error = self.critic.loss_func(prediction.flatten(), torch.FloatTensor([target]))
        if self.critic.optimizer.param_groups[0]["lr"] != 0:
            self.critic.optimizer.zero_grad()
            error.backward()
            self.critic.optimizer.step()
        return error.detach().item(), prediction.detach().item()

    def update_critic_using_buffer(self, batch_size: int = None) -> float:
        """
        batch_size < 0 : update using entire buffer (batch GD)
        batch_size > 0 : update using |batch_size| random samples from buffer (minibatch GD)
        """
        batch_size = len(self.replay_buffer) if batch_size < 0 else batch_size
        self.batch_size = batch_size
        errors = [self.err_tolerance_tau]
        #TODO in the case of minibatches, we may need to change this condition to operate
        # on moving avg rather than just the last error?
        while errors[-1] >= self.err_tolerance_tau and self.err_tolerance_tau > 0 and len(errors) < self.max_replay_steps:
            actions, targets, _ = self.replay_buffer.sample(n_samples=batch_size)
            predictions = self.critic.get_qvalues(actions)
            error = self.critic.loss_func(predictions.flatten(), torch.FloatTensor(targets))
            self.critic.optimizer_replay.zero_grad()
            error.backward()
            self.critic.optimizer_replay.step()
            errors.append(error.detach().item())
        return errors

    def update_critic_fixed_UTD(self, batch_size: int = None, UTD: int = 1) -> float:
        """
        batch_size < 0 : update using entire buffer (batch GD)
        batch_size > 0 : update using |batch_size| random samples from buffer (minibatch GD)
        UTD: how many critic updates to do
        """
        batch_size = len(self.replay_buffer) if batch_size < 0 else batch_size
        self.batch_size = batch_size
        errors = [self.err_tolerance_tau]

        eta = self.args.value_baseline_lr
        #TODO in the case of minibatches, we may need to change this condition to operate
        # on moving avg rather than just the last error?
        for i in range(UTD):
            actions, targets, _ = self.replay_buffer.sample(n_samples=batch_size)
            predictions = self.critic.get_qvalues(actions)
            error = self.critic.loss_func(predictions.flatten(), torch.FloatTensor(targets))
            self.critic.optimizer_replay.zero_grad()
            error.backward()
            self.critic.optimizer_replay.step()
            errors.append(float(error.detach().item()))
        return errors

    def update_actor(self) -> float:
        for _ in range(self.n_actor_updates):
            """
            Look at Gaussian.py for this new GaussianFixedStd policy
            Instead of rsample, it has a get_deterministic_action function
            that returns the mean.

            When interacting with the env, we use its policy.sample() to have randomness
            We should sweep over possible std_init which corresponds to randomness
            when interacting with the environment
            """
            deterministic_action = self.actor.policy.get_deterministic_action()

            qvalues = self.critic.get_qvalues(deterministic_action)
            error = - qvalues

            self.actor.optimizer.zero_grad()
            error.backward()
            self.actor.optimizer.step()

        return error.detach().item()


    def save_to_disk(self, path) -> None:
        raise NotImplementedError




