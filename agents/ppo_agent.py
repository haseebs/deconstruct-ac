import torch
import numpy as np
import sys
from copy import deepcopy
sys.path.insert(0, "../")

class PPOAgent:
    def __init__(self, args, critic, actor, replay_buffer, clip_coef, entropy_coef, v_coef, onpolicy_samples) -> None:
        self.args = args
        self.n_actor_updates = args.n_actor_updates

        self.max_replay_steps = args.max_replay_steps
        self.err_tolerance_tau = args.err_tolerance_tau

        self.actor = actor
        self.critic = critic
        self.replay_buffer = replay_buffer
        self.clip_coef = clip_coef
        self.entropy_coef = entropy_coef
        self.v_coef = v_coef
        self.batch_size = None
        self.onpolicy_samples = onpolicy_samples
        self.value_baseline = 0

        # important: linking these two config variables for simplicity
        #self.onpolicy_samples = self.n_actor_updates
        # note: previous experiments (707x) linked variables. Now we dont

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

        eta = self.args.value_baseline_lr
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
        on-policy PPO implementation
        in order to have a mini-batch of on-policy samples,
        we block the update_actor every self.onpolicy_samples steps
        to make sure on-policy samples have been stored in the buffer
        To compensate for less updates, we do n_actor_updates = onpolicy_samples

        TODO:
        we may need to consider how to replace the q value network to state value
        since PPO uses state value rather than action value
        """
        error = None
        if len(self.replay_buffer) < self.onpolicy_samples or np.mod(len(self.replay_buffer), self.onpolicy_samples)!=0:
            return 0.
        else:
            for _ in range(self.n_actor_updates * self.onpolicy_samples): # n_actor_updates will adjust update-to-data ratio
                old_actions, rewards, old_log_probs = self.replay_buffer.get_last_sample(num_samples=self.onpolicy_samples)
                rewards = torch.FloatTensor(rewards)
                old_actions = torch.FloatTensor(old_actions)
                if len(rewards) > 1:
                    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
                old_log_probs = torch.FloatTensor(old_log_probs).flatten()
                new_logprobs = self.actor.policy.log_prob(old_actions)
                with torch.no_grad():
                    advantages = rewards - self.value_baseline
                    # old_state_values = self.critic.get_qvalues(torch.FloatTensor(old_actions)).flatten()
                    state_values = torch.FloatTensor([self.value_baseline])
                    v_loss = self.critic.loss_func(state_values.squeeze(), rewards.mean())
                ratio = (new_logprobs - old_log_probs).exp()
                pg_loss1 = advantages * ratio
                pg_loss2 = advantages * torch.clamp(ratio, min=1-self.clip_coef, max=1+self.clip_coef)
                pg_loss = - torch.min(pg_loss1, pg_loss2).mean()

                _, log_probs, _, _ = self.actor.policy.rsample(num_samples=30)
                entropy_loss = -log_probs.mean()
                # entropy_loss = self.actor.policy.entropy().mean()
                error = pg_loss - self.entropy_coef * entropy_loss + self.v_coef * v_loss

                self.critic.value_net.zero_grad()
                self.actor.optimizer.zero_grad()
                error.backward()
                self.actor.optimizer.step()

            return float(error.detach().item())


    def save_to_disk(self, path) -> None:
        raise NotImplementedError




class PPOMirrorDescentAgent(PPOAgent):
    def __init__(self, args, critic, actor, replay_buffer, clip_coef, entropy_coef, v_coef, onpolicy_samples, kl_penalty_coef, md_period) -> None:
        super().__init__(args, critic, actor, replay_buffer, clip_coef, entropy_coef, v_coef, onpolicy_samples)
        self.kl_penalty_coef = kl_penalty_coef
        self.md_period = md_period
        self.frozen_policy = None
        self.count_all_actor_updates = 0

    def update_actor(self) -> float:
        """
        Algorithm 5 Mirror Descent.
        freeze the network params and compute KL penalty regularly
        KL penalty is added to the maximum likelihood loss
        """
        if len(self.replay_buffer) < self.onpolicy_samples or np.mod(len(self.replay_buffer), self.onpolicy_samples)!=0:
            return 0.
        else:
            for _ in range(self.n_actor_updates):

                if self.count_all_actor_updates % self.md_period == 0:
                    self.frozen_policy = deepcopy(self.actor.policy)
                self.count_all_actor_updates += 1

                old_actions, rewards, old_log_probs = self.replay_buffer.get_last_sample(num_samples=self.onpolicy_samples)
                rewards = torch.FloatTensor(rewards)
                old_actions = torch.FloatTensor(old_actions)

                if len(rewards) > 1:
                    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

                old_log_probs = torch.FloatTensor(old_log_probs).flatten()
                new_logprobs = self.actor.policy.log_prob(old_actions)
                with torch.no_grad():
                    advantages = rewards - self.value_baseline
                    # old_state_values = self.critic.get_qvalues(torch.FloatTensor(old_actions)).flatten()
                    state_values = torch.FloatTensor([self.value_baseline])
                    v_loss = self.critic.loss_func(state_values.squeeze(), rewards.mean())
                ratio = (new_logprobs - old_log_probs).exp()
                pg_loss1 = advantages * ratio
                pg_loss2 = advantages * torch.clamp(ratio, min=1-self.clip_coef, max=1+self.clip_coef)
                pg_loss = - torch.min(pg_loss1, pg_loss2).mean()

                sampled_actions, log_probs, _, _ = self.actor.policy.rsample(num_samples=30)
                entropy_loss = -log_probs.mean()

                mirror_descent_kl_loss = (self.actor.policy.log_prob(sampled_actions.squeeze()) - self.frozen_policy.log_prob(sampled_actions.squeeze())).mean()
                error = pg_loss - self.entropy_coef * entropy_loss + self.v_coef * v_loss - self.kl_penalty_coef * mirror_descent_kl_loss


                self.critic.value_net.zero_grad()
                self.actor.optimizer.zero_grad()
                error.backward()
                self.actor.optimizer.step()


            return float(error.detach().item())


    def update_actor_old(self) -> float:
        """
        on-policy PPO implementation
        in order to have a mini-batch of on-policy samples,
        we block the update_actor every self.onpolicy_samples steps
        to make sure on-policy samples have been stored in the buffer
        To compensate for less updates, we do n_actor_updates = onpolicy_samples

        TODO:
        we may need to consider how to replace the q value network to state value
        since PPO uses state value rather than action value
        """
        error = None
        if len(self.replay_buffer) < self.onpolicy_samples or np.mod(len(self.replay_buffer), self.onpolicy_samples)!=0:
            return 0.
        else:
            for _ in range(self.n_actor_updates):
                old_actions, rewards, old_log_probs = self.replay_buffer.get_last_sample(num_samples=self.onpolicy_samples)
                rewards = torch.FloatTensor(rewards)
                old_actions = torch.FloatTensor(old_actions)
                if len(rewards) > 1:
                    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
                old_log_probs = torch.FloatTensor(old_log_probs).flatten()
                new_logprobs = self.actor.policy.log_prob(old_actions)
                with torch.no_grad():
                    advantages = rewards - self.value_baseline
                    # old_state_values = self.critic.get_qvalues(torch.FloatTensor(old_actions)).flatten()
                    state_values = torch.FloatTensor([self.value_baseline])
                    v_loss = self.critic.loss_func(state_values.squeeze(), rewards.mean())
                ratio = (new_logprobs - old_log_probs).exp()
                pg_loss1 = advantages * ratio
                pg_loss2 = advantages * torch.clamp(ratio, min=1-self.clip_coef, max=1+self.clip_coef)
                pg_loss = - torch.min(pg_loss1, pg_loss2).mean()

                _, log_probs, _, _ = self.actor.policy.rsample(num_samples=30)
                entropy_loss = -log_probs.mean()
                # entropy_loss = self.actor.policy.entropy().mean()
                error = pg_loss - self.entropy_coef * entropy_loss + self.v_coef * v_loss

                self.critic.value_net.zero_grad()
                self.actor.optimizer.zero_grad()
                error.backward()
                self.actor.optimizer.step()

            return float(error.detach().item())
