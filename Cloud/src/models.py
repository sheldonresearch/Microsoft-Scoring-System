import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from distance.soft_dtw import SoftDTW
from data_analysis.RealData import RealData
from torch.nn.functional import softmax
from gym_hybrid.environments_beta import CreditEnv

device = torch.device('cpu')

if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.values = []
        self.rewards = []
        self.is_terminals = []
        self.actions_with_real_quota = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.actions_with_real_quota[:]
        del self.values[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()

        self.action_dim = action_dim
        self.continuous_action_var = torch.full((action_dim[1],), action_std_init * action_std_init).to(device)

        self.discrete_actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim[0]),
            nn.Softmax(dim=-1)
        )
        self.continuous_actor = nn.Sequential(
            nn.Linear(state_dim + action_dim[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim[1]),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def set_action_std(self, new_action_std):
        self.continuous_action_var = torch.full((self.action_dim[1],), new_action_std * new_action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, init_step_flag=False):
        if init_step_flag:
            discrete_action_probs = torch.tensor([1.0, 0.0, 0.0])
            discrete_dist = Categorical(discrete_action_probs)
            discrete_action = torch.tensor(0)
            discrete_action_logprob = discrete_dist.log_prob(discrete_action)
        else:
            discrete_action_probs = self.discrete_actor(state)
            discrete_dist = Categorical(discrete_action_probs)
            discrete_action = discrete_dist.sample()
            discrete_action_logprob = discrete_dist.log_prob(discrete_action)

        discrete_state = torch.cat([discrete_action_probs, state], dim=0)
        continuous_action_mean = self.continuous_actor(discrete_state)  # (-1,1)
        continuous_cov_mat = torch.diag(self.continuous_action_var).unsqueeze(dim=0)
        continuous_dist = MultivariateNormal(continuous_action_mean, continuous_cov_mat)
        continuous_action = continuous_dist.sample()
        continuous_action_logprob = continuous_dist.log_prob(continuous_action)
        return discrete_action.detach(), continuous_action.detach(), discrete_action_logprob.detach(), continuous_action_logprob.detach(), discrete_action_probs.detach()

    def evaluate(self, states, discrete_actions, continuous_actions):
        discrete_action_probs = self.discrete_actor(states)
        discrete_dist = Categorical(discrete_action_probs)
        discrete_action_logprobs = discrete_dist.log_prob(discrete_actions)
        discrete_dist_entropy = discrete_dist.entropy()

        discrete_state = torch.cat([discrete_action_probs, states], dim=1)

        continuous_action_mean = self.continuous_actor(discrete_state)
        continuous_action_var = self.continuous_action_var.expand_as(continuous_action_mean)
        continuous_cov_mat = torch.diag_embed(continuous_action_var).to(device)
        continuous_dist = MultivariateNormal(continuous_action_mean, continuous_cov_mat)
        continuous_action_logprobs = continuous_dist.log_prob(continuous_actions)

        continuous_dist_entropy = continuous_dist.entropy()

        state_values = self.critic(states)

        return discrete_action_logprobs, continuous_action_logprobs, state_values, discrete_dist_entropy, continuous_dist_entropy, discrete_dist.probs, continuous_action_mean


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=50, eps_clip=0.2
                 , continuous_action_std_init=0.6):

        self.continuous_action_std = continuous_action_std_init
        self.gamma = gamma
        self.lmbda = 0.95
        self.seperate_opti = "all_together"
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.corpus = RealData(path='../data_analysis/dataset.json')
        self.corpus_one_hot_discrete_actions = self.corpus.one_hot_discrete_actions
        self.corpus_parameters = self.corpus.parameters

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, continuous_action_std_init).to(device)

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.discrete_actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.continuous_actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.distance_optimizer = torch.optim.Adam([
            {'params': self.policy.discrete_actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.continuous_actor.parameters(), 'lr': lr_actor}
        ])

        self.discrete_actor_optimizer = torch.optim.Adam(
            [{'params': self.policy.discrete_actor.parameters(), 'lr': lr_actor}])
        self.continuous_actor_optimizer = torch.optim.Adam(
            [{'params': self.policy.continuous_actor.parameters(), 'lr': lr_actor}])

        self.actor_optimizer = torch.optim.Adam([
            {'params': self.policy.discrete_actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.continuous_actor.parameters(), 'lr': lr_actor}
        ])
        self.critic_optimizer = torch.optim.Adam([{'params': self.policy.critic.parameters(), 'lr': lr_critic}])

        self.policy_old = ActorCritic(state_dim, action_dim, continuous_action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        self.continuous_action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def get_dist_by_quota(self, discrete_dist_probs, continuous_action_mean):
        criterion = SoftDTW(gamma=1.0, normalize=True)

        d2 = torch.unsqueeze(discrete_dist_probs, 0)
        dy = self.corpus_one_hot_discrete_actions
        d3 = d2.expand(dy.shape[0], d2.shape[1], d2.shape[2])
        dis_loss = criterion(d3, dy).mean()

        c2 = torch.unsqueeze(continuous_action_mean, 0)
        cy = self.corpus_parameters
        c3 = c2.expand(cy.shape[0], c2.shape[1], c2.shape[2])
        con_loss = criterion(c3, cy).mean()
        loss = 0.0001 * dis_loss + 0.0001 * con_loss

        return loss

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.continuous_action_std = self.continuous_action_std - action_std_decay_rate
        self.continuous_action_std = round(self.continuous_action_std, 4)
        if self.continuous_action_std <= min_action_std:
            self.continuous_action_std = min_action_std
        self.set_action_std(self.continuous_action_std)

    def select_action(self, state, init_step_flag=False):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            discrete_action, continuous_action, discrete_action_logprob, continuous_action_logprob, discrete_action_probs = self.policy_old.act(
                state, init_step_flag=init_step_flag)

        action = (discrete_action, continuous_action)
        action_logprob = (discrete_action_logprob, continuous_action_logprob)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return discrete_action.item(), continuous_action.detach().cpu().numpy().flatten()

    def get_advantages(self, values, rewards, is_terminals):

        returns = []
        gae = 0
        last_value = values[-1]

        for i in reversed(range(len(rewards))):
            if is_terminals[i]:
                mask = 0
            else:
                mask = 1

            last_value = last_value * mask

            delta = rewards[i] + self.gamma * last_value - values[i]
            gae = mask * gae
            gae = delta + self.gamma * self.lmbda * gae

            last_value = values[i]

            returns.insert(0, gae + values[i])

        adv = torch.tensor(returns) - values
        rewards_to_go = torch.tensor(returns, dtype=torch.float32).to(device)

        return rewards_to_go, (adv - adv.mean()) / (adv.std() + 1e-10)

    def update(self, mode='gae'):

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_discrete_actions = torch.squeeze(torch.stack([act[0] for act in self.buffer.actions], dim=0)).detach().to(
            device)
        old_discrete_logprobs = torch.squeeze(
            torch.stack([prob[0] for prob in self.buffer.logprobs], dim=0)).detach().to(device)

        old_continuous_actions = torch.squeeze(torch.stack([act[1] for act in self.buffer.actions], dim=0)).detach().to(
            device)
        old_continuous_logprobs = torch.squeeze(
            torch.stack([prob[1] for prob in self.buffer.logprobs], dim=0)).detach().to(device)


        rewards_to_go = []
        if mode == 'monte_carlo':
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards_to_go.insert(0, discounted_reward)


            rewards_to_go = torch.tensor(rewards_to_go, dtype=torch.float32).to(device)
            rewards_to_go = (rewards_to_go - rewards_to_go.mean()) / (rewards_to_go.std() + 1e-7)


        for _ in range(self.K_epochs):

            discrete_action_logprobs, \
            continuous_action_logprobs, \
            state_values, \
            discrete_dist_entropy, \
            continuous_dist_entropy, \
            discrete_dist_probs, \
            continuous_action_mean = self.policy.evaluate(old_states,
                                                          old_discrete_actions,
                                                          old_continuous_actions)
            state_values = torch.squeeze(state_values)


            discrete_ratios = torch.exp(discrete_action_logprobs - old_discrete_logprobs.detach())
            continuous_ratios = torch.exp(continuous_action_logprobs - old_continuous_logprobs.detach())

            advantages = 0
            if mode == 'monte_carlo':
                advantages = rewards_to_go - state_values.detach()
            elif mode == 'gae':
                rewards_to_go, advantages = self.get_advantages(state_values.detach(), self.buffer.rewards,
                                                                self.buffer.is_terminals)

            discrete_surr1 = discrete_ratios * advantages
            discrete_surr2 = torch.clamp(discrete_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            continuous_surr1 = continuous_ratios * advantages
            continuous_surr2 = torch.clamp(continuous_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages


            critic_discount = 0.01
            entropy_discount = 0.005
            attacker_loss_dis = -torch.min(discrete_surr1,
                                           discrete_surr2) - entropy_discount * discrete_dist_entropy

            attacker_loss_con = - torch.min(continuous_surr1,
                                            continuous_surr2) - entropy_discount * continuous_dist_entropy


            actor_loss = attacker_loss_dis + attacker_loss_con

            critic_loss = self.MseLoss(state_values, rewards_to_go)


            if self.seperate_opti == "seperate_actor_critic":
                self.actor_optimizer.zero_grad()
                actor_loss.mean().backward()
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
            elif self.seperate_opti == "seperate_discrete_continue_critic":
                if _ == 0:
                    self.distance_optimizer.zero_grad()
                    dis_loss = self.get_dist_by_quota(discrete_dist_probs=discrete_dist_probs,
                                                      continuous_action_mean=continuous_action_mean)

                    dis_loss.backward(retain_graph=True)
                    self.distance_optimizer.step()
                else:

                    self.continuous_actor_optimizer.zero_grad()
                    attacker_loss_con.mean().backward(retain_graph=True)

                    self.continuous_actor_optimizer.step()

                    self.discrete_actor_optimizer.zero_grad()
                    attacker_loss_dis.mean().backward()
                    self.discrete_actor_optimizer.step()

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()
                    self.critic_optimizer.zero_grad()

            elif self.seperate_opti == "all_together":
                if _ == 0:
                    dis_loss = self.get_dist_by_quota(discrete_dist_probs=discrete_dist_probs,
                                                      continuous_action_mean=continuous_action_mean)
                    loss = actor_loss + critic_discount * critic_loss + 0.01 * dis_loss
                else:
                    loss = actor_loss + critic_discount * critic_loss

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

class GradientBooster(nn.Module):
    def __init__(self, init_weight=None,
                 init_decay=None,
                 init_tanh_scale=None,
                 lr=0.01):
        super(GradientBooster, self).__init__()
        self.env_agent = CreditEnv(mode='test')
        self.weight = nn.Parameter(torch.empty((5, 1)))
        self.decay = nn.Parameter(torch.empty((6, 1)))
        self.tanh_scale = nn.Parameter(torch.empty((7, 1)))

        self.init_parameter(init_weight, init_decay, init_tanh_scale)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=0.01, weight_decay=0.0001)

    def init_parameter(self, init_weight, init_decay, init_tanh_scale):
        if init_weight is None:
            nn.init.uniform_(self.weight.data, a=0, b=1)
        else:
            self.weight.data = init_weight
        if init_decay is None:
            nn.init.uniform_(self.init_decay.data, a=0, b=1)
        else:
            self.decay.data = init_decay
        if init_tanh_scale is None:
            nn.init.uniform_(self.init_tanh_scale.data, a=0, b=1)
        else:
            self.tanh_scale.data = init_tanh_scale

    def reward_loss(self, test_trajectories=None, env_details=None):
        weight = softmax(self.weight, dim=0)
        decay = torch.clip(self.decay, min=0.0001, max=1.0)
        tanh_scale = torch.clip(self.tanh_scale, min=0.0001, max=10)
        self.env_agent.set_weights(w=weight, decay=decay, tanh_scale=tanh_scale)

        test_running_reward = 0.0
        entry_num = 0
        for trajectory, detail in zip(test_trajectories, env_details):
            self.env_agent.set_meta(detail)
            ep_reward = 0.0
            for action in trajectory:
                entry_num = entry_num + 1
                action = (int(action[0]), torch.squeeze(action[1]))
                reward = self.env_agent.step(action)
                ep_reward = ep_reward + reward
            test_running_reward = test_running_reward + ep_reward
        if len(test_trajectories) == 0:
            print("alert!")
        avg_test_reward = test_running_reward

        return avg_test_reward
