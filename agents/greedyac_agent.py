import torch
import numpy as np
import sys
sys.path.insert(0, "../")
from simulators.vectorized_pid_controller import VectorizedPIDController
from .base_agent import BaseAgent
from copy import deepcopy



# i think we can combine some of these AC agents later
class GreedyACAgent:
    def __init__(self, args, critic, proposal_actor, behavior_actor, replay_buffer, use_true_reward=False) -> None:
        self.args = args
        self.n_actor_updates = args.n_actor_updates
        self.n_action_proposals = args.n_action_proposals
        self.n_action_proposals_selected = args.n_action_proposals_selected

        self.max_replay_steps = args.max_replay_steps
        self.err_tolerance_tau = args.err_tolerance_tau

        self.proposal_actor = proposal_actor
        self.behavior_actor = behavior_actor
        self.critic = critic
        self.replay_buffer = replay_buffer
        self.value_baseline = 0
        self.simulator = VectorizedPIDController()
        self.use_true_reward = use_true_reward

    def update_value_baseline(self, reward: float) -> None:
        return

    @torch.no_grad()
    def act(self) -> list[float]:
        action, log_probs, _, _ = self.behavior_actor.policy.sample(num_samples=1)
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
        # TODO maybe something like do_KL can be argument here. The KL computation
        # can be a part of the policy class
        behavior_error = None
        proposal_error = None
        for _ in range(self.n_actor_updates):
            # sample |n_action_proposals| actions from proposal policy
            sampled_actions = self.proposal_actor.policy.sample(num_samples=self.n_action_proposals)[0].detach()
            # get qvalues for the sampled actions
            if self.use_true_reward:
                qvalues = torch.FloatTensor(self.simulator.step(sampled_actions.detach().cpu().numpy())).view(-1, 1)
            else:
                qvalues = self.critic.get_qvalues(sampled_actions)

            # get top |n_action_proposals_selected| actions according to qvalues
            sorted_q = torch.argsort(qvalues, dim=0, descending=True)

            best_indices = sorted_q[:2*self.n_action_proposals_selected]
            best_actions_for_behavior = sampled_actions[best_indices[:self.n_action_proposals_selected]].squeeze()
            best_actions_for_proposal = sampled_actions[best_indices].squeeze()

            behavior_error = - self.behavior_actor.policy.log_prob(best_actions_for_behavior).mean()
            proposal_error = - self.proposal_actor.policy.log_prob(best_actions_for_proposal).mean()

            """actor update, paper Alg.4 line 14"""
            self.behavior_actor.optimizer.zero_grad()
            self.critic.value_net.zero_grad()
            behavior_error.backward()
            self.behavior_actor.optimizer.step()

            """proposal update"""
            self.proposal_actor.optimizer.zero_grad()
            self.critic.value_net.zero_grad()
            proposal_error.backward()
            self.proposal_actor.optimizer.step()

        return float(behavior_error.detach().item())


class GreedyACMirrorDescentAgent(GreedyACAgent):
    def __init__(self, args, critic, proposal_actor, behavior_actor, replay_buffer, kl_penalty_coef, md_period) -> None:
        super().__init__(args, critic, proposal_actor, behavior_actor, replay_buffer)
        self.kl_penalty_coef = kl_penalty_coef
        self.md_period = md_period
        self.frozen_proposal_policy = None
        self.frozen_behavior_policy = None
        self.count_all_actor_updates = 0

    def update_actor(self) -> float:
        """
        Algorithm 5 Mirror Descent.
        freeze the network params and compute KL penalty regularly
        KL penalty is added to the maximum likelihood loss
        """
        behavior_error = None
        proposal_error = None
        for _ in range(self.n_actor_updates):
            if self.count_all_actor_updates % self.md_period == 0:
                self.frozen_proposal_policy = deepcopy(self.proposal_actor.policy)
                self.frozen_behavior_policy = deepcopy(self.behavior_actor.policy)
            self.count_all_actor_updates += 1
            # sample |n_action_proposals| actions from proposal policy
            sampled_actions = self.proposal_actor.policy.sample(num_samples=self.n_action_proposals)[0].detach()
            # get qvalues for the sampled actions
            qvalues = self.critic.get_qvalues(sampled_actions)

            # get top |n_action_proposals_selected| actions according to qvalues
            sorted_q = torch.argsort(qvalues, dim=0, descending=True)

            best_indices = sorted_q[:2*self.n_action_proposals_selected]
            best_actions_for_behavior = sampled_actions[best_indices[:self.n_action_proposals_selected]].squeeze()
            best_actions_for_proposal = sampled_actions[best_indices].squeeze()

            kl_behavior = (self.behavior_actor.policy.log_prob(sampled_actions.squeeze()) - self.frozen_behavior_policy.log_prob(sampled_actions.squeeze())).mean()
            kl_proposal = (self.proposal_actor.policy.log_prob(sampled_actions.squeeze()) - self.frozen_proposal_policy.log_prob(sampled_actions.squeeze())).mean()

            behavior_error = - self.behavior_actor.policy.log_prob(best_actions_for_behavior).mean() + self.kl_penalty_coef * kl_behavior
            proposal_error = - self.proposal_actor.policy.log_prob(best_actions_for_proposal).mean() + self.kl_penalty_coef * kl_proposal

            """actor update, paper Alg.4 line 14"""
            self.behavior_actor.optimizer.zero_grad()
            self.critic.value_net.zero_grad()
            behavior_error.backward()
            self.behavior_actor.optimizer.step()

            """proposal update"""
            self.proposal_actor.optimizer.zero_grad()
            self.critic.value_net.zero_grad()
            proposal_error.backward()
            self.proposal_actor.optimizer.step()

        return float(behavior_error.detach().item())

    def save_to_disk(self, path) -> None:
        raise NotImplementedError
