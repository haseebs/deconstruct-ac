import torch
import numpy as np
import sys
sys.path.insert(0, "../")

class VMPOAgent:
    def __init__(self, args, critic, actor, replay_buffer, eps_clip=0.2) -> None:
        self.args = args
        self.n_actor_updates = args.n_actor_updates
        self.n_action_proposals = args.n_action_proposals
        self.n_action_proposals_selected = args.n_action_proposals_selected

        self.max_replay_steps = args.max_replay_steps
        self.err_tolerance_tau = args.err_tolerance_tau

        self.actor = actor
        self.critic = critic
        self.replay_buffer = replay_buffer
        self.eps_clip = eps_clip
        self.eta = torch.autograd.Variable(torch.tensor(1.0), requires_grad=True)
        self.alpha = torch.autograd.Variable(torch.tensor(0.1), requires_grad=True)
        self.eps_eta = 0.02
        self.eps_alpha = 0.1
        self.clip_coef = 0.2
        self.batch_size = None
        self.value_baseline = 0

    def update_value_baseline(self, reward: float) -> None:
        return

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
        batch_size = len(self.replay_buffer) if self.batch_size is None else self.batch_size

        for _ in range(self.n_actor_updates):
            """
            MPO: maximum posterior optimization, a trajectory based alg that explicitly considers KL as a penalty to reward
            VMPO: an online version of MPO that is more suitable for deep RL

            KL policy takes the form \pi_t \propto \pi_{t-1} \exp(Advantage / temperature)

            This implementation adapts VMPO, see the paper here: https://arxiv.org/pdf/1909.12238.pdf
            Note that the original MPO is tied to Gaussian policy and decoupling them is not easy.
            I referred to the implementation here: https://github.com/YYCAAA/V-MPO_Lunarlander/blob/main/VMPO.py#L66
            Since VMPO and PPO are both online algs, they basically share the logic flow

            A: advantage function, in our bandit version this becomes r_t - V
            value_loss: r_t - V; notice that
            eta, alpha: temperature coefficients, the Lagrangian multipliers to be optimized
            For more info see the PPO implementation
            """
            old_actions, rewards, old_log_probs = self.replay_buffer.sample(n_samples=batch_size)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            old_log_probs = torch.FloatTensor(old_log_probs).squeeze(-1)
            old_qvalues = self.critic.get_qvalues(torch.FloatTensor(old_actions))
            advantages = torch.FloatTensor(rewards) - old_qvalues.detach().flatten()

            actions, new_logprobs, _, _ = self.actor.policy.rsample(num_samples=batch_size)
            q_values = self.critic.get_qvalues(actions).flatten()

            new_probs = (new_logprobs.exp() / new_logprobs.exp().sum()).detach()
            old_probs = old_log_probs.exp() / old_log_probs.exp().sum()
            is_ratio = torch.clamp(new_probs / old_probs, 1 - self.clip_coef, 1 + self.clip_coef)

            v_value = (q_values.flatten() * new_probs).sum()
            weighted_r = (torch.FloatTensor(rewards) * is_ratio).mean()
            v_loss = self.critic.loss_func(v_value, weighted_r)

            """use only advantage and log probabilities from good performing runs"""
            advprobs = torch.stack((advantages,new_logprobs))
            advprobs = advprobs[:,torch.sort(advprobs[0],descending=True).indices]

            good_advantages = advprobs[0,:len(old_log_probs)//2]
            good_logprobs = advprobs[1,:len(old_log_probs)//2]

            phis = torch.exp(good_advantages/self.eta.detach()) / torch.sum(torch.exp(good_advantages / self.eta.detach()))
            L_pi = -phis * good_logprobs
            L_eta = self.eta * self.eps_eta + self.eta * torch.log(torch.mean(torch.exp(good_advantages / self.eta)))

            kl = self.compute_kl(old_log_probs.exp(), old_log_probs,  new_logprobs)

            L_alpha = torch.mean(self.alpha * (self.eps_alpha - kl.detach()) + self.alpha.detach() * kl)

            error = L_pi.mean() + L_eta + L_alpha + 0.5 * v_loss

            self.actor.optimizer.zero_grad()
            error.backward()
            self.actor.optimizer.step()

        return float(error.detach().item())

    def compute_kl(self, p1, logp1, logp2):
        """make sure you have this renormalization step"""
        dim = 1 if len(p1.shape) > 1 else 0
        p1 = p1 / p1.sum(dim=dim, keepdim=True)
        return (p1 * (logp1 - logp2)).sum(dim=dim, keepdim=True)

    def save_to_disk(self, path) -> None:
        raise NotImplementedError




