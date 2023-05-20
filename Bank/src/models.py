import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from distance.soft_dtw import SoftDTW
from data_analysis.RealData import RealData
# import numpy as np
# from torch.nn.functional import tanh
from torch.nn.functional import softmax
from gym_hybrid.environments_beta import CreditEnv
# import pickle

################################## set device ##################################

print("============================================================================================")

# set device to cpu or cuda
device = torch.device('cpu')

if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

print("============================================================================================")


################################## PPO Policy ##################################

class RolloutBuffer:
    def __init__(self):
        self.actions = []  # [(dis_act_id,[continuous_para1,continuous_para2])]
        self.states = []
        self.logprobs = []  # [(discrete_action_logprob, continuous_action_logprob)]
        # self.discrete_action_probs = []
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
        # del self.discrete_action_probs[:]
        del self.actions_with_real_quota[:]
        del self.values[:]


class ActorCritic(nn.Module):  # in the near future, this class will be renamed as Attacker
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()

        self.action_dim = action_dim  # [discrete_action_dim,continuous_action_dim]
        self.continuous_action_var = torch.full((action_dim[1],), action_std_init * action_std_init).to(device)

        # actor
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
            # later we wish change state_dim as: state_dim+ action_dim[0] or else.
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim[1]),
            nn.Tanh()  # (-1,1)
        )

        """ """
        # critic
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

        # in the near future, the self.continuous_actor() should take encoder(cat(state,discrete_dist)) as input
        # or the self.continuous_actor() only takes discrete_dist as input
        # continuous_action_mean =0.5*(self.continuous_actor(state) +1)#(-1,1)--->(0,1)

        discrete_state = torch.cat([discrete_action_probs, state], dim=0)

        """dim=0 because:
        discrete_action_probs.shape:    torch.Size([2])
        states.shape:                   torch.Size([9])
        discrete_state.shape:           torch.Size([11]) 
        """

        continuous_action_mean = self.continuous_actor(discrete_state)  # (-1,1)
        continuous_cov_mat = torch.diag(self.continuous_action_var).unsqueeze(dim=0)
        continuous_dist = MultivariateNormal(continuous_action_mean, continuous_cov_mat)
        continuous_action = continuous_dist.sample()
        # print("final continuous_action: ",continuous_action)
        continuous_action_logprob = continuous_dist.log_prob(continuous_action)
        # print(continuous_action_logprob)

        return discrete_action.detach(), continuous_action.detach(), discrete_action_logprob.detach(), continuous_action_logprob.detach(), discrete_action_probs.detach()

    def evaluate(self, states, discrete_actions, continuous_actions):  # , discrete_action_probs):
        discrete_action_probs = self.discrete_actor(states)
        discrete_dist = Categorical(discrete_action_probs)
        discrete_action_logprobs = discrete_dist.log_prob(discrete_actions)
        discrete_dist_entropy = discrete_dist.entropy()

        discrete_state = torch.cat([discrete_action_probs, states], dim=1)
        """dim=1 because:
        discrete_action_probs.shape:    torch.Size([1000, 2])
        states.shape:                   torch.Size([1000, 9])
        discrete_state.shape:           torch.Size([1000, 11])
        """
        continuous_action_mean = self.continuous_actor(discrete_state)
        continuous_action_var = self.continuous_action_var.expand_as(continuous_action_mean)
        continuous_cov_mat = torch.diag_embed(continuous_action_var).to(device)
        continuous_dist = MultivariateNormal(continuous_action_mean, continuous_cov_mat)
        continuous_action_logprobs = continuous_dist.log_prob(continuous_actions)

        # continuous_action = continuous_dist.sample()
        # print("continuous_action_mean: ",continuous_action_mean,continuous_action_mean.requires_grad)
        # print("discrete_dist.probs: ",discrete_dist.probs,discrete_dist.probs.requires_grad)

        continuous_dist_entropy = continuous_dist.entropy()
        # # For Single Action Environments. for example, only one coninue action
        # if self.action_dim == 1:
        #     action = action.reshape(-1, self.action_dim)

        # print("hahahahahah77777")
        # print(discrete_action_logprobs) # 训练集动作在新参数下的概率分布
        # print(discrete_action_probs) # 训练数据在新的参数下的离散动作概率，可以直接用这个当作预测结果来计算与真实数据的距离
        """
        tensor([[0.4516, 0.5484],
        [0.4535, 0.5465],
        [0.4535, 0.5465],
        ...,
        [0.4536, 0.5464],
        [0.4536, 0.5464],
        [0.4536, 0.5464]], grad_fn=<SoftmaxBackward0>)
        """
        # print(discrete_dist) # 这个东西可以看作是读取训练集后的预测输出，可以直接用这个来计算与real data的距离
        # print(discrete_dist.probs)
        # print(discrete_dist.logits)
        # print(continuous_dist)
        # dis_actions= discrete_dist.sample()
        # con_actions=continuous_dist.sample()
        # print(dis_actions)
        # print(dis_actions.shape)
        # print(con_actions)
        # print(con_actions.shape)

        state_values = self.critic(states)

        return discrete_action_logprobs, continuous_action_logprobs, state_values, discrete_dist_entropy, continuous_dist_entropy, discrete_dist.probs, continuous_action_mean


class PPO:  # in the near future, this class will be renamed as AttackerOptimizer
    def __init__(self, state_dim, action_dim, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=50, eps_clip=0.2
                 , continuous_action_std_init=0.6):
        """
        :param state_dim:
        :param action_dim: action_dim=[discrete_action_dim,continuous_action_dim]
        :param lr_actor:
        :param lr_critic:
        :param gamma:
        :param K_epochs:
        :param eps_clip:
        :param has_continuous_action_space:
        :param action_std_init:
        """

        self.continuous_action_std = continuous_action_std_init
        self.gamma = gamma
        self.lmbda = 0.95  # GAE的折扣因子，默认值是 0.98 （可选0.96 ~ 0.99）lambda越小：方差越小但偏差越大
        self.seperate_opti ="only_simu"#"only_attack"# "seperate_discrete_continue_critic"  # "seperate_discrete_continue_critic"  # "seperate_actor_critic" # "all_together", "seperate_discrete_continue_critic"
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        # self.corpus = RealData(path='../data_analysis/351globalid.json', sampled_n=10)
        self.corpus =RealData(path='../data_analysis/bill_detail_test.txt', sampled_n=10)
        self.corpus_one_hot_discrete_actions = self.corpus.one_hot_discrete_actions
        self.corpus_parameters = self.corpus.parameters

        self.buffer = RolloutBuffer()
        """
            self.actions = []  # [(dis_act_id,[continuous_para1,continuous_para2])]
            self.states = []
            self.logprobs = []
            self.rewards = []
            self.is_terminals = []
        """

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
        """
        this function is used in attacker loss

        Method 1：
        初步构想：距离包含两个项目（1）离散模式（2）quota序列

        离散模式：根据新的discrete_dist_probs，计算其与真实数据的one_hot_discrete_actions的距离

        quota序列：
        - 根据过去数据的动作类型重新计算对应的para：得到新的continuous_action (action.parameter)
        - 在buffer中记录每个动作对应的min_ask,max_ask,min_dep,max_dep
        - 根据buffer中的min,max，重新构建出新的quota序列
        - 计算这个quota序列和真实序列的相似性

        Method 2：
        仅仅计算离散模式


        Method 3：
        仅仅计算宏观统计量


        Method 4[recommend!]：
        离散模式和Method1相同，
        quota序列这里，想办法把真实数据的quota序列映射到parameters中：
        对于一条序列：
        找到min ask quota, max ask quota, min deploy quota, 各个时间点的max available quota（数据集天然携带）
        当前如果是ask q, parameter=(q-min_ask)/(max-min)*2-1
        当前如果是dep q, parameter=(q-min_dep)/(max available quota at t-min deploy)*2-1

        """
        # actions=torch.unsqueeze(actions,dim=1)
        criterion = SoftDTW(gamma=1.0, normalize=True)

        # discrete distance
        d2 = torch.unsqueeze(discrete_dist_probs, 0)
        dy = self.corpus_one_hot_discrete_actions
        d3 = d2.expand(dy.shape[0], d2.shape[1], d2.shape[2])



        dis_loss = criterion(d3, dy).mean()  # Just like any torch.nn.xyzLoss()

        # continuous distance
        c2 = torch.unsqueeze(continuous_action_mean, 0)
        cy = self.corpus_parameters



        c3 = c2.expand(cy.shape[0], c2.shape[1], c2.shape[2])
        con_loss = criterion(c3, cy).mean()  # Just like any torch.nn.xyzLoss()

        # print("dis_loss_mean: ",dis_loss_mean,"one_hot_discrete_actions len: ",len(one_hot_discrete_actions)," | con_loss_mean: ",con_loss_mean, "continuous_action_parameters len: ",len(continuous_action_parameters))
        loss = 0.0001 * dis_loss + 0.0001 * con_loss

        return loss  # torch.tanh((dis_loss_mean+con_loss_mean / 500))  # dis_loss.mean()

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        self.continuous_action_std = self.continuous_action_std - action_std_decay_rate
        self.continuous_action_std = round(self.continuous_action_std, 4)
        if self.continuous_action_std <= min_action_std:
            self.continuous_action_std = min_action_std
            print("setting actor output action_std to min_action_std : ", self.continuous_action_std)
        else:
            print("setting actor output action_std to : ", self.continuous_action_std)
        self.set_action_std(self.continuous_action_std)

        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state, init_step_flag=False):
        # discrete action
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            discrete_action, continuous_action, discrete_action_logprob, continuous_action_logprob, discrete_action_probs = self.policy_old.act(
                state, init_step_flag=init_step_flag)

        action = (discrete_action, continuous_action)
        action_logprob = (discrete_action_logprob, continuous_action_logprob)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        # self.buffer.discrete_action_probs.append(discrete_action_probs)
        return discrete_action.item(), continuous_action.detach().cpu().numpy().flatten()

    def get_advantages(self, values, rewards, is_terminals):
        """
        https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-2-2-gae-and-ppo-loss-fe1b3c5549e8
        https://nn.labml.ai/rl/ppo/gae.html
            rewards: self.buffer.rewards
            is_terminals: self.buffer.is_terminals
        """
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
        """
        mode='gae','monte_carlo'
        """
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_discrete_actions = torch.squeeze(torch.stack([act[0] for act in self.buffer.actions], dim=0)).detach().to(
            device)
        old_discrete_logprobs = torch.squeeze(
            torch.stack([prob[0] for prob in self.buffer.logprobs], dim=0)).detach().to(device)

        old_continuous_actions = torch.squeeze(torch.stack([act[1] for act in self.buffer.actions], dim=0)).detach().to(
            device)
        old_continuous_logprobs = torch.squeeze(
            torch.stack([prob[1] for prob in self.buffer.logprobs], dim=0)).detach().to(device)

        """
        https://zhuanlan.zhihu.com/p/345687962
        强化学习中值函数与优势函数的估计方法
        """
        # rewards to go
        rewards_to_go = []  # reward to go
        if mode == 'monte_carlo':
            # Monte Carlo estimate of returns
            discounted_reward = 0
            """
                self.actions = []  # [(dis_act_id,[continuous_para1,continuous_para2])]
                self.states = []
                self.logprobs = []  # [(discrete_action_logprob, continuous_action_logprob)]
                self.rewards = []
                self.is_terminals = []
            """
            for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards_to_go.insert(0, discounted_reward)

            # Normalizing the rewards
            """
                 这个地方写论文的时候记得提一下，我觉得这个地方就是解决所谓偏移问题的方法
            """
            rewards_to_go = torch.tensor(rewards_to_go, dtype=torch.float32).to(device)
            rewards_to_go = (rewards_to_go - rewards_to_go.mean()) / (rewards_to_go.std() + 1e-7)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):  # here is just equal to our previous algorithm's attackerOptimizer
            """
            在标准的PPO训练算法中，这里的K_epochs其实相当于1
            https://blog.csdn.net/melody_cjw/article/details/112971830
            """
            # Evaluating old actions and values
            # logprobs, state_values, dist_entropy
            discrete_action_logprobs, \
            continuous_action_logprobs, \
            state_values, \
            discrete_dist_entropy, \
            continuous_dist_entropy, \
            discrete_dist_probs, continuous_action_mean = self.policy.evaluate(old_states,
                                                                               old_discrete_actions,
                                                                               old_continuous_actions)  # , old_discrete_action_probs)

            if self.seperate_opti == 'only_simu':
                loss = self.get_dist_by_quota(discrete_dist_probs=discrete_dist_probs,
                                              continuous_action_mean=continuous_action_mean)
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
            else:

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)

                # Finding the ratio (pi_theta / pi_theta__old)
                discrete_ratios = torch.exp(discrete_action_logprobs - old_discrete_logprobs.detach())
                continuous_ratios = torch.exp(continuous_action_logprobs - old_continuous_logprobs.detach())
                """
                https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-3-4-82081ea58146
                """

                # Finding Surrogate Loss
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

                """
                final loss of clipped objective PPO
                you will find the answers to all your confused thins here: 
                https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-2-2-gae-and-ppo-loss-fe1b3c5549e8
                """
                critic_discount = 0.01
                entropy_discount = 0.005  # 它越大，会让新旧策略越不同。一般为0.01，可选0.005~0.05

                # if we want to optimize discrete actor, continuous actor, and critic joinly
                attacker_loss_dis = -torch.min(discrete_surr1,
                                               discrete_surr2) - entropy_discount * discrete_dist_entropy

                attacker_loss_con = - torch.min(continuous_surr1,
                                                continuous_surr2) - entropy_discount * continuous_dist_entropy

                # attacker_loss = torch.relu(attacker_loss)
                actor_loss = attacker_loss_dis + attacker_loss_con

                critic_loss = self.MseLoss(state_values, rewards_to_go)
                # critic_loss =torch.relu(critic_loss)

                # print("attacker_loss_dis: {} "
                #       "attacker_loss_con: {} "
                #       "critic_loss: {}".format(attacker_loss_dis.mean().item(),
                #                                attacker_loss_con.mean().item(),
                #                                critic_loss.item()))

                """
                因为你的模型里dis的输出放进了con的输入里，所以无法独立backward con loss和dis loss
                想要独立优化dis和con，你就需要拆解dis和con为两个独立的component，像那篇论文那样，可以试试看效果如何
                seperate_discrete_continue_critic
                """

                # "seperate_actor_critic" # "all_together", "seperate_discrete_continue_critic"
                if self.seperate_opti == "seperate_actor_critic":
                    self.actor_optimizer.zero_grad()
                    actor_loss.mean().backward()
                    self.actor_optimizer.step()
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()
                elif self.seperate_opti == "seperate_discrete_continue_critic":

                    """
                    假如你既想让discrete的输出作为continue的输入，同时又想让他们两个用两个独立的优化器独立更新，那么也许你可以这样做：
                    先backward continue loss，更新continue actor，由于continue的optimizer里只注册了continue层的参数，因此不会
                    更新discrete的参数。但是传导到discrete的gradient会被保留
    
                    加下来再调用一次continue optimizer的zero_grad()，清除continue optimizer传导的gradient
    
                    再开始dicrete的optimizer
    
                    """

                    if _ == 0:
                        self.distance_optimizer.zero_grad()
                        dis_loss = self.get_dist_by_quota(discrete_dist_probs=discrete_dist_probs,
                                                          continuous_action_mean=continuous_action_mean)

                        dis_loss.backward(retain_graph=True)
                        self.distance_optimizer.step()
                    else:

                        self.continuous_actor_optimizer.zero_grad()
                        attacker_loss_con.mean().backward(retain_graph=True)
                        """FYI:
                        https://blog.csdn.net/weixin_44058333/article/details/99701876
                        """
                        self.continuous_actor_optimizer.step()

                        self.discrete_actor_optimizer.zero_grad()
                        attacker_loss_dis.mean().backward()
                        self.discrete_actor_optimizer.step()

                        self.critic_optimizer.zero_grad()
                        critic_loss.backward()
                        self.critic_optimizer.step()
                        self.critic_optimizer.zero_grad()

                elif self.seperate_opti == "only_attack":
                    loss = actor_loss + critic_discount * critic_loss
                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    self.optimizer.step()

                elif self.seperate_opti == "all_together":

                    if _ == 0:
                        dis_loss = self.get_dist_by_quota(discrete_dist_probs=discrete_dist_probs,
                                                          continuous_action_mean=continuous_action_mean)
                        loss = actor_loss + critic_discount * critic_loss + 0.01 * dis_loss
                    else:
                        loss = actor_loss + critic_discount * critic_loss

                    # # test
                    # loss=self.get_dist_by_quota(discrete_dist_probs=discrete_dist_probs,
                    #                                       continuous_action_mean=continuous_action_mean)

                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    self.optimizer.step()
            """
            according to this
            https://spinningup.openai.com/en/latest/algorithms/ppo.html#id2
            it updates critic separately
            """

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


################################## Booster Policy ##################################
class GradientBooster(nn.Module):
    def __init__(self, init_weight=None,  # torch.tensor([[1],[1],[1],[1],[1]],dtype=torch.float32),
                 init_decay=None,  # torch.tensor([[0.05], [0.05], [0.05], [0.05], [0.05], [1]],dtype=torch.float32),
                 init_tanh_scale=None,
                 lr=0.01,
                 # torch.tensor([[0.0001], [0.0001], [0.01], [0.0001], [0.0001], [0.001], [1]],dtype=torch.float32),
                 real_data=None):
        super(GradientBooster, self).__init__()
        self.env_agent = CreditEnv(mode='test')
        # self.env_agent2=gym_hybrid.environments.CreditEnv(mode='test')
        self.weight = nn.Parameter(torch.empty((5, 1)))
        self.decay = nn.Parameter(torch.empty((6, 1)))
        self.tanh_scale = nn.Parameter(torch.empty((7, 1)))

        self.init_parameter(init_weight, init_decay, init_tanh_scale)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=0.01, weight_decay=0.0001)
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        # self.corpus = RealData(path='../data_analysis/351globalid.json').one_hot_discrete_actions
        # self.corpus = np.random.choice(self.corpus, size=int(len(self.corpus) * 0.1))
        # self.benchmark_data=real_data
        # self.soft_dtw=SoftDTW()

    # def distance_loss(self,test_trajectories):
    #     return self.soft_dtw(test_trajectories,self.benchmark_data)

    def init_parameter(self, init_weight, init_decay, init_tanh_scale):
        if init_weight is None:
            # print("xixi")
            nn.init.uniform_(self.weight.data, a=0, b=1)
        else:
            self.weight.data = init_weight

        if init_decay is None:
            # print("xixi")
            nn.init.uniform_(self.init_decay.data, a=0, b=1)
        else:
            self.decay.data = init_decay

        if init_tanh_scale is None:
            # print("xixi")
            nn.init.uniform_(self.init_tanh_scale.data, a=0, b=1)
        else:
            self.tanh_scale.data = init_tanh_scale

    def real_data_scores(self):
        """
        here we read real data corpus, and repeat the credit to obtain scores for the sequences
        so that we can calculate the distance between real data and generated data
        """
        pass

    def dis_loss(self):
        pass

    def reward_loss(self, test_trajectories=None, env_details=None):  # , weights=None):
        # print("hahah",self.weight)
        weight = softmax(self.weight, dim=0)
        # print(weight)
        # decay=relu(self.decay)
        # tanh_scale=relu(self.tanh_scale)
        decay = torch.clip(self.decay, min=0.0001, max=1.0)  # relu(gb.decay) #建议把所有的relu改为clamp (0.0001,gb.decay)
        tanh_scale = torch.clip(self.tanh_scale, min=0.0001, max=10)  # relu(gb.tanh_scale)
        # all_scores=[]
        self.env_agent.set_weights(w=weight, decay=decay, tanh_scale=tanh_scale)

        # self.env_agent2.set_weights(w=weight,decay=self.decay,tanh_scale=self.tanh_scale)
        test_running_reward = 0.0  # torch.tensor(0.0,requires_grad=True)#0  # torch.tensor(0,requires_grad=True)
        entry_num = 0
        for trajectory, detail in zip(test_trajectories, env_details):
            self.env_agent.set_meta(detail)
            # self.env_agent2.set_meta(detail)
            ep_reward = 0.0
            # ep_scores=[]
            for action in trajectory:
                # print(action)
                entry_num = entry_num + 1
                action = (int(action[0]), torch.squeeze(action[1]))
                reward = self.env_agent.step(action)  # 如果ini_max_avaialbe_quota需要处理，这里也需要修改
                # reward = self.env_agent2.step(action)

                # a_scores=_["states"]
                # action_scores=a_scores[2:]
                # ep_scores.append(action_scores)
                ep_reward = ep_reward + reward
            test_running_reward = test_running_reward + ep_reward
            # all_scores.append(ep_scores)
        if len(test_trajectories) == 0:
            print("alert!")
        avg_test_reward = test_running_reward  # /len(test_trajectories)# entry_num
        """
        note that (100.0*100)=max_ep_len*total_test_episodes in the generate_corpus function
        """
        # all_scores=np.array(all_scores,dtype=np.float)
        return avg_test_reward  # all_scores

    # def total_loss(self,test_trajectories,env_details):
    #     return self.distance_loss()+self.reward_loss()
