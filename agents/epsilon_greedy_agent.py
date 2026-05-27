import torch
import numpy as np
from .base_agent import BaseAgent


class EpsilonGreedyAgent(BaseAgent):
    def __init__(self, args, value_func, loss_func, optimizer, replay_buffer) -> None:
        super(EpsilonGreedyAgent, self).__init__(args)
        self.epsilon: float = args.epsilon_starting
        self.epsilon_ending: float = args.epsilon_ending
        self.epsilon_decay_amount: float = args.epsilon_decay_amount
        self.epsilon_decay_every: int = args.epsilon_decay_every

        self.n_actions: int = args.n_actions
        self.feature_dim: int = args.feature_dim
        self.n_bins: int = args.n_bins

        self.action_min: float = args.action_min
        self.action_max: float = args.action_max

        self.max_replay_steps: int = args.max_replay_steps
        self.err_tolerance_tau: float = args.err_tolerance_tau

        self.value_func = value_func
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer

    def __anneal_epsilon(self, step) -> None:
        # assumes the act() is called at every step
        if step % self.epsilon_decay_every == 0:
            self.epsilon -= self.epsilon_decay_amount
        if self.epsilon < self.epsilon_ending:
            self.epsilon = self.epsilon_ending

    def act(self, step) -> list[float]:
        self.__anneal_epsilon(step)
        with torch.no_grad():
            if np.random.random() > (1.0 - self.epsilon):
                action = np.random.random(self.n_actions) * self.action_max
            else:
                all_action_vals = self.value_func.forward_all_features()
                action = self.value_func.get_actions_from_idx(torch.argmax(all_action_vals))
        return action.tolist()

    def update(self, action: list[float], target: float) -> float:
        self.replay_buffer.append(action, target, None)
        prediction = self.value_func([action])
        error = self.loss_func(prediction, torch.FloatTensor([target]))
        self.optimizer.zero_grad()
        error.backward()
        self.optimizer.step()
        return error.detach().item(), prediction.detach().item()

    def update_using_batch_gd(self) -> list[float]:
        errors = [self.err_tolerance_tau]
        while errors[-1] >= self.err_tolerance_tau and self.err_tolerance_tau > 0 and len(errors) < self.max_replay_steps:
            actions, targets, _ = self.replay_buffer.get_all_samples()
            predictions = self.value_func(actions)
            error = self.loss_func(predictions.flatten(), torch.FloatTensor(targets))
            self.optimizer.zero_grad()
            error.backward()
            self.optimizer.step()
            errors.append(error.detach().item())
        return errors

    def save_to_disk(self, path) -> None:
        raise NotImplementedError
