import torch
import numpy as np
import sys
from copy import deepcopy
sys.path.insert(0, "../")
from .base_agent import BaseAgent
from simulators.vectorized_pid_controller import VectorizedPIDController


# i think we can combine some of these AC agents later
class MPOAdaptiveAgent:
    def __init__(self, args, critic, actor, replay_buffer, kl_coef, use_true_reward = False) -> None:
        self.args = args
        self.n_actor_updates = args.n_actor_updates
        self.n_action_proposals = args.n_action_proposals
        self.n_action_proposals_selected = args.n_action_proposals_selected

        self.max_replay_steps = args.max_replay_steps
        self.err_tolerance_tau = args.err_tolerance_tau

        self.actor = actor
        self.critic = critic
        self.replay_buffer = replay_buffer
        self.value_baseline = 0
        self.simulator = VectorizedPIDController()
        self.use_true_reward = use_true_reward
        self.temperature = kl_coef
        self.alpha = getattr(args, 'alpha_init', 1.0)
        self.alpha_lr = getattr(args, 'alpha_lr', 0.01)
        self.eps_kl = getattr(args, 'eps_kl', 0.01)
        self.frozen_policy = None

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
        """
        Faithful MPO actor update with adaptive KL trust region.

        min_theta E_{a~pi_t}[-exp(q(a)/tau) * ln pi_theta(a)]
                  - alpha * E_{a~pi_t}[ln pi_t(a) - ln pi_theta(a)]

        alpha update: alpha -= alpha_lr * (eps_kl - mean_kl), clamp >= 0

        where pi_t is the frozen old policy, refreshed each call.
        """
        # Freeze the old policy at the start of each update cycle
        self.frozen_policy = deepcopy(self.actor.policy)

        actor_loss = None
        for _ in range(self.n_actor_updates):
            # Sample actions from the FROZEN old policy pi_t
            with torch.no_grad():
                sampled_actions = self.frozen_policy.sample(num_samples=self.n_action_proposals)[0]

            # Get Q-values for the sampled actions
            if self.use_true_reward:
                true_reward = self.simulator.step(sampled_actions.detach().cpu().numpy())
                qvalues = torch.FloatTensor(true_reward)
            else:
                qvalues = self.critic.get_qvalues(sampled_actions.detach())
            qvalues = qvalues - qvalues.max()

            # Exponential advantage weights (detached — no gradient through Q-values)
            exp_scale = torch.exp((qvalues.squeeze() - self.value_baseline) / self.temperature).detach()

            # Log-probs under current policy pi_theta (gradient flows here)
            log_probs_theta = self.actor.policy.log_prob(sampled_actions)

            # Log-probs under frozen policy pi_t (no gradient)
            with torch.no_grad():
                log_probs_frozen = self.frozen_policy.log_prob(sampled_actions)

            # Weighted MLE loss: -E_{pi_t}[exp((q-v)/tau) * ln pi_theta(a)]
            mle_loss = -torch.mean(exp_scale * log_probs_theta)

            # Per-sample KL: ln(pi_t(a)/pi_theta(a)) = ln pi_t(a) - ln pi_theta(a)
            per_sample_kl = log_probs_frozen - log_probs_theta
            kl_penalty = -self.alpha * per_sample_kl.mean()

            # Total actor loss
            actor_loss = mle_loss + kl_penalty

            # Dual variable update: alpha -= alpha_lr * (eps - mean_kl), clamp >= 0
            mean_kl = per_sample_kl.mean().detach().item()
            self.alpha -= self.alpha_lr * (self.eps_kl - mean_kl)
            self.alpha = max(self.alpha, 0.0)

            self.actor.optimizer.zero_grad()
            self.critic.value_net.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

        return float(actor_loss.detach().item())


    def save_to_disk(self, path) -> None:
        raise NotImplementedError
