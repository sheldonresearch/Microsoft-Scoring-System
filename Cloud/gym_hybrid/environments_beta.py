import numpy as np
from typing import Tuple
from typing import Optional
import gym
from gym import spaces
from gym.utils import seeding
import torch
from torch.distributions import Normal

gym.logger.set_level(40)


class Action:
    def __init__(self, id_: int, parameters: list):
        self.id = id_
        self.parameters = parameters

    @property
    def parameter(self) -> float:
        return self.parameters[self.id]


class CreditEnv(gym.Env):
    def __init__(self, seed: Optional[int] = None, mode='run'):
        self.mode = mode
        self.max_step = None
        self.history = None

        self.max_available_quota = None
        self.min_deploy = None
        self.min_ask = None
        self.max_ask = None

        self.traceback_window = None
        self.ask_fre_score = None
        self.dep_fre_score = None
        self.ask_score = None
        self.deploy_score = None
        self.uti_score = None
        self.last_score = None
        self.current_score = None
        self.growth_score = None
        self.wait_score = None
        self.reward = None
        self.current_step = None

        self.w = None
        self.decay = None
        self.tanh_scale = None
        self.historical_max_deploy_quota = 0

        self.uti_score_his = []
        self.deploy_quota_records = []
        self.deploy_parameter_records = []
        self.ask_quota_records = []
        self.max_available_quota_records = []

        if self.mode == 'run':
            self.seed(seed)
            self.reset()
            parameters_min = np.array([0, 0, 0])
            parameters_max = np.array([1000, 1000, 0.0001])
            self.action_space = spaces.Tuple((spaces.Discrete(3),
                                              spaces.Box(parameters_min, parameters_max)))

            observation_min = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            observation_max = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            self.observation_space = spaces.Box(observation_min, observation_max)

    def seed(self, seed: Optional[int] = None) -> list:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_meta(self, info=None):
        self.max_step = info["max_step"]
        self.history = []

        self.max_available_quota = info["max_available_quota"]
        self.min_deploy = info["min_deploy"]
        self.min_ask = info["min_ask"]
        self.max_ask = info["max_ask"]

        self.traceback_window = info["traceback_window"]
        self.ask_fre_score = info["ask_fre_score"]
        self.dep_fre_score = info["dep_fre_score"]
        self.ask_score = info["ask_score"]
        self.deploy_score = info["deploy_score"]
        self.uti_score = info["uti_score"]
        self.growth_score = info["growth_score"]
        self.last_score = info["last_score"]
        self.current_score = info["current_score"]
        self.current_step = info["current_step"]

        self.uti_score_his = []
        self.deploy_quota_records = []
        self.ask_quota_records = []
        self.historical_max_deploy_quota = 0
        self.deploy_parameter_records = []

    def set_weights(self, w=None, decay=None, tanh_scale=None):
        self.w = w
        self.decay = decay
        self.tanh_scale = tanh_scale

    def get_states(self):
        if self.mode == 'run':
            states = [
                self.max_available_quota / 10000,
                self.current_step / self.max_step,
                self.ask_fre_score,
                self.dep_fre_score,
                self.ask_score,
                self.deploy_score,
                self.uti_score,
                self.growth_score,
                self.last_score,
                self.current_score
            ]
            return states
        else:
            raise ValueError("get_states() is not available when self.mode is '{}'".format(self.mode))

    def reset(self) -> list:
        if self.mode == 'run':
            self.history = []
            self.traceback_window = 30
            self.max_step = 100
            self.max_available_quota = Normal(torch.tensor([2000.0]), torch.tensor([100.0])).sample().item()
            self.ask_fre_score = 0.0
            self.dep_fre_score = 0.0
            self.ask_score = 0.0
            self.deploy_score = 0.0
            self.uti_score = 0.0
            self.last_score = 0.0
            self.current_score = 0.0
            self.growth_score = 0.0
            self.reward = 0
            self.current_step = 0
            self.min_ask, self.max_ask = 1000, 10000
            self.min_deploy = 1000
            self.uti_score_his = []
            self.deploy_quota_records = []
            self.ask_quota_records = []
            self.historical_max_deploy_quota = 0
            self.deploy_parameter_records = []
            return self.get_states()
        else:
            raise ValueError(
                "reset() is not available when self.mode is '{}'".format(self.mode))

    def get_details(self):
        if self.mode == 'run':
            return {
                "max_step": self.max_step,
                "history": self.history,
                "max_available_quota": self.max_available_quota,
                "min_deploy": self.min_deploy,
                "min_ask": self.min_ask,
                "max_ask": self.max_ask,
                "traceback_window": self.traceback_window,
                "ask_fre_score": self.ask_fre_score,
                "dep_fre_score": self.dep_fre_score,
                "ask_score": self.ask_score,
                "deploy_score": self.deploy_score,
                "uti_score": self.uti_score,
                "growth_score": self.growth_score,
                # "wait_score": self.wait_score,
                "last_score": self.last_score,
                "current_score": self.current_score,
                "reward": self.reward,
                "current_step": self.current_step}
        else:
            raise ValueError("get_details() is not available when self.mode is '{}'".format(self.mode))

    def quota(self, par, min_quota, max_quota):
        par = 0.5 * (par + 1)
        _quota = par * (max_quota - min_quota) + min_quota
        _quota = np.clip(_quota, a_min=min_quota, a_max=max_quota)
        return _quota

    def tanh(self, val):
        if self.mode == 'run':
            return np.tanh(val)
        return torch.tanh(val)

    def abs(self, val):
        if self.mode == 'run':
            return np.abs(val)
        return torch.abs(val)

    def _temporal_difference(self, records: list) -> torch.Tensor:
        seq = [q for q in records if q > 0]
        seq_shift = [0] + seq
        seq.append(0)
        seq_td = (np.array(seq) - np.array(seq_shift))[1:-1]
        a = torch.tensor(seq_td)
        a = torch.reshape(a, (-1, 1))
        return a

    def _decay_sum2(self, records, decay, condition_fun=None, exter_fun=None):
        weighted_sum = 0.0
        decay_items = []
        for ii, cha in enumerate(records):
            t = len(records) - ii - 1
            if self.mode == 'run':
                dec = np.exp(-decay * t)
            elif self.mode == 'test':
                dec = torch.exp(-decay * t)

            if isinstance(cha, Action):
                exter_fun_input = cha.parameter
                condition_fun_input = cha.id
            else:
                exter_fun_input = cha
                condition_fun_input = None

            if condition_fun(condition_fun_input):
                q = exter_fun(exter_fun_input)
                weighted_sum = weighted_sum + dec * q
                decay_items.append(dec * q)
            else:
                decay_items.append(0)

        return weighted_sum, decay_items

    def _update_ask_score(self, ):

        weighted_ave_quota, decay_items = self._decay_sum2(records=self.ask_quota_records,
                                                           decay=self.decay[0],
                                                           condition_fun=lambda x: True,
                                                           exter_fun=lambda x: x)

        regular, _ = self._decay_sum2(records=self.ask_quota_records,
                                      decay=self.decay[0],
                                      condition_fun=lambda x: True,
                                      exter_fun=lambda x: self.max_ask)

        self.ask_score =self.tanh(-self.tanh_scale[0] * weighted_ave_quota / regular) + 1
        return decay_items

    def _update_deploy_score(self):
        weighted_ave_quota, decay_items = self._decay_sum2(records=self.deploy_quota_records,
                                                           decay=self.decay[1],
                                                           condition_fun=lambda x: True,
                                                           exter_fun=lambda x: x)

        regular, _ = self._decay_sum2(records=self.max_available_quota_records,
                                      decay=self.decay[1],
                                      condition_fun=lambda x: True,
                                      exter_fun=lambda x: x)

        self.deploy_score = self.tanh(self.tanh_scale[1] * weighted_ave_quota / regular)
        return decay_items

    def _meta_util_score(self):
        if len(self.deploy_quota_records) == 0:
            deploy_quota = 0.0
        else:
            deploy_quota = self.deploy_quota_records[-1]
        uti = deploy_quota / self.max_available_quota
        self.uti_score_his.append(uti)
        if len(self.uti_score_his) > self.traceback_window:
            self.uti_score_his = self.uti_score_his[len(self.uti_score_his) - self.traceback_window:]

    def _update_util_score(self, ):
        self._meta_util_score()
        weighted_ave_uti, decay_items = self._decay_sum2(records=self.uti_score_his, decay=self.decay[2],
                                                         condition_fun=lambda x: True,
                                                         exter_fun=lambda x: x)

        regular, _ = self._decay_sum2(records=self.uti_score_his, decay=self.decay[2],
                                      condition_fun=lambda x: True,
                                      exter_fun=lambda x: 1)
        if regular == 0:
            regular = 1
        self.uti_score = self.tanh(self.tanh_scale[2] * weighted_ave_uti / regular)
        return decay_items

    def _update_ask_fre_score(self, ):
        weighted_ask_num, decay_items = self._decay_sum2(records=self.history,
                                                         decay=self.decay[3],
                                                         condition_fun=lambda x: x == 0,
                                                         exter_fun=lambda q: 1)

        regular, _ = self._decay_sum2(records=self.history,
                                      decay=self.decay[3],
                                      condition_fun=lambda x: True,
                                      exter_fun=lambda q: 1)
        if regular == 0:
            regular = 1
        self.ask_fre_score = self.tanh(self.tanh_scale[3] * weighted_ask_num / regular)
        return decay_items

    def _update_dep_fre_score(self, ):
        weighted_dep_num, decay_items = self._decay_sum2(records=self.history,
                                                         decay=self.decay[4],
                                                         condition_fun=lambda x: x == 1,
                                                         exter_fun=lambda q: 1)

        regular, _ = self._decay_sum2(records=self.history,
                                      decay=self.decay[4],
                                      condition_fun=lambda x: True,
                                      exter_fun=lambda q: 1)
        if regular == 0:
            regular = 1
        self.dep_fre_score = self.tanh(self.tanh_scale[4] * weighted_dep_num / regular)
        return decay_items

    def _update_growth_score(self, ):
        deploy_changes = self._temporal_difference(records=self.deploy_quota_records)
        ask_changes = self._temporal_difference(records=self.ask_quota_records)

        deploy_changes = torch.flatten(deploy_changes).detach().numpy()
        ask_changes = torch.flatten(ask_changes).detach().numpy()

        if len(deploy_changes) == 0 and len(ask_changes) == 0:
            self.growth_score = 1.0
        else:
            dep_coe = len(deploy_changes) * 1.0 / (len(deploy_changes) + len(ask_changes))
            ask_coe = 1 - dep_coe

            weighted_dep_change, _ = self._decay_sum2(records=deploy_changes, decay=self.decay[5],
                                                      condition_fun=lambda x: True, exter_fun=lambda x: np.abs(x))

            weighted_ask_change, _ = self._decay_sum2(records=ask_changes, decay=self.decay[5],
                                                      condition_fun=lambda x: True, exter_fun=lambda x: np.abs(x))

            weighted_change = (dep_coe * weighted_dep_change + ask_coe * weighted_ask_change)/100000
            self.growth_score = self.tanh(-self.tanh_scale[5] * weighted_change) + 1

    def _update_wait_score(self):

        weighted_wait_num, decay_items = self._decay_sum2(records=self.history, decay=self.decay[6],
                                                          condition_fun=lambda x: x == 2, exter_fun=lambda q: 1)

        regular, _ = self._decay_sum2(records=self.history, decay=self.decay[6],
                                      condition_fun=lambda x: True, exter_fun=lambda q: 1)

        if regular == 0:
            regular = 1

        if self.uti_score > 0.9:
            self.wait_score = self.tanh(self.tanh_scale[6] * weighted_wait_num / regular)
        else:
            self.wait_score = self.tanh(-self.tanh_scale[6] * weighted_wait_num / regular) + 1

        return decay_items

    def update_scores(self):

        util_score_decay_items = self._update_util_score()
        ask_fre_decay_items = self._update_ask_fre_score()
        dep_fre_decay_items = self._update_dep_fre_score()
        dep_score_decay_items = self._update_deploy_score()
        ask_score_decay_items = self._update_ask_score()
        self._update_growth_score()
        self.current_score = self.w[0] * self.ask_score + self.w[1] * self.deploy_score + self.w[2] * self.uti_score + \
                             self.w[3] * self.ask_fre_score + self.w[4] * self.dep_fre_score + self.w[
                                 5] * self.growth_score

        self.current_score = self.tanh(self.tanh_scale[7] * self.current_score)

        return util_score_decay_items, ask_fre_decay_items, dep_fre_decay_items, dep_score_decay_items, ask_score_decay_items

    def get_reward(self, action):

        if self.mode == 'test':
            r = torch.tensor(0.0, requires_grad=True)
        else:
            r = 0

        branch = 0
        delta_s = self.tanh(self.abs(self.current_score - self.last_score)) * 10  # 10
        assert (action.id in [0, 1, 2]), "illegal action id: {}".format(action.id)
        if action.id == 1:
            quota = self.quota(action.parameter, min_quota=self.min_deploy, max_quota=self.max_available_quota)
            if quota > self.historical_max_deploy_quota:
                if self.current_score > self.last_score:
                    r = r - 0.01
                    branch = 1
                else:
                    r = 1.2 * delta_s
                    branch = 2
            else:
                if quota > self.deploy_quota_records[-2]:
                    if self.current_score > self.last_score:
                        r = r - 0.02
                        branch = 3
                    else:
                        r = 1.1 * delta_s
                        branch = 4
                else:
                    if self.current_score < self.last_score:
                        r = r - 0.02
                        branch = 5
                    else:
                        r = delta_s
                        branch = 6
        elif action.id == 0:
            if self.current_score < self.last_score:
                r = r - 0.06
                branch = 7
            elif self.current_score > self.last_score:
                r = delta_s
                branch = 8
        elif action.id == 2:
            r = r - 0.04
            branch = 9

        return r, branch

    def step(self, raw_action: Tuple[int, list], init_step_flag=False) -> Tuple[list, float, bool, dict]:
        action = Action(*raw_action)
        deploy_quota = 0.0
        ask_quota = 0.0
        deploy_parameter = 0.0
        if action.id == 0:
            ask_quota = self.quota(action.parameter, min_quota=self.min_ask, max_quota=self.max_ask)
            if init_step_flag:
                ask_quota = self.max_available_quota
            if len(self.deploy_quota_records) == 0:
                deploy_quota = 0
                deploy_parameter = 0.0
            else:
                deploy_quota = self.deploy_quota_records[-1]
                deploy_parameter = self.deploy_parameter_records[-1]

            self.max_available_quota = self.max_available_quota + ask_quota
        elif action.id == 1:
            deploy_quota = self.quota(action.parameter, min_quota=self.min_deploy, max_quota=self.max_available_quota)
            ask_quota = 0.0
            deploy_parameter = action.parameter
        elif action.id == 2:
            ask_quota = 0.0
            deploy_quota = 0.0
            deploy_parameter = 0.0

        self.ask_quota_records.append(ask_quota)
        self.deploy_quota_records.append(deploy_quota)
        self.deploy_parameter_records.append(deploy_parameter)
        self.max_available_quota_records.append(self.max_available_quota)
        self.history.append(action)

        if len(self.history) > self.traceback_window:
            self.history = self.history[len(self.history) - self.traceback_window:]
            self.ask_quota_records = self.ask_quota_records[len(self.ask_quota_records) - self.traceback_window:]
            self.deploy_quota_records = self.deploy_quota_records[
                                        len(self.deploy_quota_records) - self.traceback_window:]
            self.deploy_parameter_records = self.deploy_parameter_records[
                                            len(self.deploy_parameter_records) - self.traceback_window:]

        self.current_step += 1

        util_score_decay_items, ask_fre_decay_items, dep_fre_decay_items, dep_score_decay_items, ask_score_decay_items = self.update_scores()

        if self.mode == 'run':
            if self.current_step >= self.max_step:
                done = True
            else:
                done = False
            self.reward, branch = self.get_reward(action)
            self.historical_max_deploy_quota = max(self.historical_max_deploy_quota, deploy_quota)

            states = self.get_states()
            self.last_score = self.current_score
            return states, self.reward, done, {'branch': branch,
                                               'historical_max_deploy_quota': self.historical_max_deploy_quota,
                                               'deploy_quota': deploy_quota,
                                               'render': {
                                                   'weight_factor_score': [self.w[0] * self.ask_score,
                                                                           self.w[1] * self.deploy_score,
                                                                           self.w[2] * self.uti_score,
                                                                           self.w[3] * self.ask_fre_score,
                                                                           self.w[4] * self.dep_fre_score,
                                                                           self.w[5] * self.growth_score],
                                                   'util_score_decay_items': util_score_decay_items,
                                                   'ask_fre_decay_items': ask_fre_decay_items,
                                                   'dep_fre_decay_items': dep_fre_decay_items,
                                                   'dep_score_decay_items': dep_score_decay_items,
                                                   'ask_score_decay_items': ask_score_decay_items}
                                               }
        elif self.mode == 'test':
            self.reward, branch = self.get_reward(action)
            self.historical_max_deploy_quota = max(self.historical_max_deploy_quota, deploy_quota)
            self.last_score = self.current_score
            return self.reward


