import torch
from torch.nn.functional import softmax



class GBA(object):
    def __init__(self):
        """gradient_booster_attacker.py attacker_train()"""

        self.max_ep_len = 30 # max episode length in one episode
        self.episode_number = 500#5000  # 2000
        self.max_training_timesteps = self.episode_number * self.max_ep_len  # 3000000  # break training loop if timeteps > max_training_timesteps
        self.print_freq = self.max_ep_len * 2  # print avg reward in the interval (in num timesteps)
        self.log_freq = self.max_ep_len * 2  # log avg reward in the interval (in num timesteps)
        self.save_model_freq = self.max_ep_len * 10  # int(1e5)  # save model frequency (in num timesteps)

        self.action_std = 0.6  # starting std for action distribution (Multivariate Normal)
        self.action_std_decay_rate = 0.1  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
        self.min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
        self.action_std_decay_freq = self.max_ep_len * 100  # int(2.5e5)  # action_std decay frequency (in num timesteps)

        self.update_timestep = self.max_ep_len * 2  # update policy every n timesteps
        self.K_epochs = 10  # 20  # update policy for K epochs in one PPO update

        self.eps_clip = 0.2  # clip parameter for PPO 它越小，表示信任域越窄，策略更新越谨慎，从避免让新旧策略差异过大 默认0.2，可选0.1-0.3
        self.gamma = 0.98  # discount factor

        self.lr_actor = 0.0001  # learning rate for actor network
        self.lr_critic = 0.001  # learning rate for critic network

        """gradient_booster_attacker.py if __name__ == '__main__':"""
        self.booster_epoch = 5
        self.max_version = 20
        self.booster_lr=0.001

        weight = torch.tensor([[1.0],  # ask_score
                               [1.0],  # deploy_score
                               [1.0],  # trade_off_score
                               [1.0]]) # wait_score


        weight = softmax(weight, dim=0)

        decay = torch.tensor([[0.05],  # ask_score
                              [0.05],  # deploy_score
                              [0.05],  # trade_off_score
                              [0.05]])# wait_score

        tanh_scale = torch.tensor([[0.7],  # ask_score
                                   [0.7],  # deploy_score
                                   [0.7],  # trade_off_score
                                   [0.7],  # wait_score
                                   [0.7]]) # total

        self.ini_credit_parameters = {
            'for_v':0,
            'weight': weight,
            'decay': decay,
            'tanh_scale': tanh_scale
        }

        """generate_corpus"""
        self.test_max_ep_len = 40
        self.total_test_episodes = 10 # total num of testing episodes


class Draw(object):
    def __init__(self):
        self.min_window_len_smooth = 1
        self.window_len_smooth = 20
        self.ci_train = 95
        self.ci_test = 65
        self.max_ep_len = 40
        self.total_test_episodes = 50


class Config(object):
    def __init__(self):
        self.gba = GBA()
        self.draw = Draw()


if __name__ == '__main__':
    c = Config()
    print(c.gba.gamma)
