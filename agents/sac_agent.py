import torch
import numpy as np
import sys
sys.path.insert(0, "../")
# from simulators.pid_controller import PIDController
from simulators.vectorized_pid_controller import VectorizedPIDController
from .base_agent import BaseAgent
from copy import deepcopy


# i think we can combine some of these AC agents later
class SACAgent:
    def __init__(self, args, critic, actor, replay_buffer, entropy_coef=0.01, baseline=False, use_reparameterisation_trick=False, use_true_reward=False) -> None:
        self.args = args
        self.n_actor_updates = args.n_actor_updates
        self.n_action_proposals = args.n_action_proposals
        self.n_action_proposals_selected = args.n_action_proposals_selected

        self.max_replay_steps = args.max_replay_steps
        self.err_tolerance_tau = args.err_tolerance_tau

        self.actor = actor
        self.critic = critic
        self.replay_buffer = replay_buffer
        self.entropy_coef = entropy_coef
        self.use_baseline = baseline
        self.batch_size = None
        self.value_baseline = 0
        self.use_reparameterisation_trick = use_reparameterisation_trick
        self.simulator = VectorizedPIDController()
        self.use_true_reward = use_true_reward

    def update_value_baseline(self, reward: float) -> None:
        eta = self.args.value_baseline_lr
        self.value_baseline = ((1 - eta) * self.value_baseline) + (eta * reward)

    @torch.no_grad()
    def act(self) -> list[float]:
        action, log_probs, _, _ = self.actor.policy.sample(num_samples=1)
        return action.tolist(), log_probs.tolist()

    def update_critic(self, action: list[float], target: float, log_probs: list[float]) -> float:
        self.replay_buffer.append(action, target, log_probs)
        prediction = self.critic.get_qvalues([action])
        error = self.critic.loss_func(prediction.flatten(), torch.FloatTensor([target]))
        if self.critic.optimizer.param_groups[0]["lr"] != 0:
            self.critic.optimizer.zero_grad()
            error.backward()
            self.critic.optimizer.step()
        return float(error.detach().item()), float(prediction.detach().item())

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
            errors.append(float(error.detach().item()))
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
        error = None
        for _ in range(self.n_actor_updates):
            """
            reference of sac loss
            https://github.com/samuelfneumann/GreedyAC/blob/5ebce50ba3d55d68896fe82dc5b1c970ac588fff/agent/nonlinear/SAC.py#L282C36-L282C36
            """
            if self.use_reparameterisation_trick:
                sampled_actions, log_probs, _, _ = self.actor.policy.rsample(num_samples=self.n_action_proposals)
                error = self.actor_loss_reparameterisation_trick(sampled_actions, log_probs)
            else:
                sampled_actions, log_probs, _, _ = self.actor.policy.sample(num_samples=self.n_action_proposals)
                """flag to use true reward here"""
                if self.use_true_reward:
                    error = self.actor_loss_log_liklihood_true_reward(sampled_actions, log_probs)
                else:
                    error = self.actor_loss_log_liklihood(sampled_actions, log_probs)

            self.actor.optimizer.zero_grad()
            self.critic.value_net.zero_grad()
            error.backward()
            self.actor.optimizer.step()
        return float(error.detach().item())

    def actor_loss_reparameterisation_trick(self, sampled_actions, log_probs) -> torch.tensor:
        qvalues = self.critic.get_qvalues(sampled_actions)
        qvalues -= self.value_baseline
        error = -(qvalues.flatten() - self.entropy_coef * log_probs).mean()
        return error

    def actor_loss_log_liklihood_old(self, sampled_actions, log_probs) -> torch.tensor:
        probs = log_probs.exp()
        with torch.no_grad():
            qvalues = self.critic.get_qvalues(sampled_actions)
            qvalues -= self.value_baseline
            scale = qvalues - self.entropy_coef * log_probs.unsqueeze(-1)
        error = -(probs.squeeze() * scale.squeeze()).mean()
        return error


    def actor_loss_log_liklihood(self, sampled_actions, log_probs) -> torch.tensor:
        with torch.no_grad():
            qvalues = self.critic.get_qvalues(sampled_actions).view(-1)
            scale = qvalues - self.value_baseline - self.entropy_coef * log_probs.view(-1)
        error = -(scale.squeeze() * log_probs.squeeze()).mean()
        return error

    def actor_loss_log_liklihood_true_reward(self, sampled_actions, log_probs) -> torch.tensor:
        probs = log_probs
        with torch.no_grad():
            """call vectorized simulator to output a vector of rewards"""
            true_reward = self.simulator.step(sampled_actions.detach().cpu().numpy())
            true_reward = torch.FloatTensor(true_reward)
            scale = true_reward - self.value_baseline - self.entropy_coef * log_probs
        error = -(probs.squeeze() * scale.squeeze()).mean()
        return error

    def save_to_disk(self, path) -> None:
        raise NotImplementedError



class SACMirrorDescentAgent(SACAgent):
    def __init__(self, args, critic, actor, replay_buffer, entropy_coef=0.01, baseline=False, use_reparameterisation_trick=False, kl_penalty_coef=1.0, md_period=10) -> None:
        super().__init__(args, critic, actor, replay_buffer, entropy_coef, baseline, use_reparameterisation_trick)
        self.kl_penalty_coef = kl_penalty_coef
        self.md_period = md_period
        self.frozen_policy = None
        self.count_all_actor_updates = 0

    def update_actor(self) -> float:
        error = None
        for _ in range(self.n_actor_updates):

            if self.count_all_actor_updates % self.md_period == 0:
                self.frozen_policy = deepcopy(self.actor.policy)
            self.count_all_actor_updates += 1

            sampled_actions, log_probs, _, _ = self.actor.policy.rsample(num_samples=self.n_action_proposals)
            if self.use_reparameterisation_trick:
                error = self.actor_loss_reparameterisation_trick(sampled_actions, log_probs)
            else:
                error = self.actor_loss_log_liklihood(sampled_actions, log_probs)

            mirror_descent_kl_loss = (self.actor.policy.log_prob(sampled_actions.squeeze()) - self.frozen_policy.log_prob(sampled_actions.squeeze())).mean()
            error -= self.kl_penalty_coef * mirror_descent_kl_loss

            self.actor.optimizer.zero_grad()
            self.critic.value_net.zero_grad()
            error.backward()
            self.actor.optimizer.step()
        return float(error.detach().item())
