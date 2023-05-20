import torch
from torch.nn.functional import softmax



class GBA(object):
    def __init__(self):
        self.max_ep_len = 100
        self.episode_number = 5000
        self.max_training_timesteps = self.episode_number * self.max_ep_len
        self.print_freq = self.max_ep_len * 2
        self.log_freq = self.max_ep_len * 2
        self.save_model_freq = self.max_ep_len * 10
        self.action_std = 0.6
        self.action_std_decay_rate = 0.1
        self.min_action_std = 0.1
        self.action_std_decay_freq = self.max_ep_len * 100

        self.update_timestep = self.max_ep_len * 10
        self.K_epochs = 5

        self.eps_clip = 0.2
        self.gamma = 0.98

        self.lr_actor = 0.0001
        self.lr_critic = 0.001

        self.booster_epoch = 5
        self.max_version = 20
        self.booster_lr=0.001

        weight = torch.tensor([[1.0],
                               [1.0],
                               [1.0],
                               [1.0],
                               [1.0],
                               [1.0]])
        weight = softmax(weight, dim=0)

        decay = torch.tensor([[0.05],
                              [0.05],
                              [0.05],
                              [0.05],
                              [0.05],
                              [0.05],
                              [0.05]
                              ])

        tanh_scale = torch.tensor([[0.7],
                                   [0.7],
                                   [0.7],
                                   [0.7],
                                   [0.7],
                                   [0.7],
                                   [0.7],
                                   [0.7]
                                   ])

        self.ini_credit_parameters = {
            'for_v':0,
            'weight': weight,
            'decay': decay,
            'tanh_scale': tanh_scale
        }

        self.test_max_ep_len = 100
        self.total_test_episodes = 10


class Config(object):
    def __init__(self):
        self.gba = GBA()


