import numpy as np
from typing import Tuple
from typing import Optional
import gym
from gym import spaces
from gym.utils import seeding
import torch
from torch.distributions import Normal  # , Uniform

# from distance.soft_dtw import SoftDTW

"""
这个文件致力于探索environment较大幅度的改进，包括重用等等
待测试稳定后，再转移到environments.py上
"""
gym.logger.set_level(40)  # noqa


class Action:
    """"
    Action class to store and standardize the action for the environment.
    """

    def __init__(self, id_: int, parameters: list):
        """"
        Initialization of an action.

        Args:
            id_: The id of the selected action.
            parameters: The parameters of an action.
        """
        self.id = id_
        self.parameters = parameters

    @property
    def parameter(self) -> float:
        """

                 26703
                 00:740--->wait: 上期账单金额,上期还款金额:希望越少越好：因为这意味着用户不活跃了:action_type=3
                 0x:426--->ask: 上期账单金额为0,上期还款金额:希望还款金额越多越好：因为这意味着用户有较强的还款能力: action_type=0
                 x0:1785-->deploy：上期账单金额,上期还款金额为0:希望上期账单金额越少越好：因为这意味着用户有少的负债: action_type=1
                 xx:23011-->trade-off:上期账单金额,上期还款金额:希望两者比值接近1：因为这意味着用户可以即时还清债务并且现金流充分: action_type=2

              """
        if self.id in [0, 1]:
            return self.parameters[self.id]
        elif self.id == 2:
            return self.parameters
        elif self.id == 3:
            return 0.0


class CreditEnv(gym.Env):
    def __init__(self, seed: Optional[int] = None, mode='run'):
        """
        mode='run', by default
        mode='test', only for testing
        """
        # self.seed=0
        self.mode = mode
        self.max_step = None
        self.history = None

        self.max_available_quota = None
        self.min_deploy = None
        self.min_ask = None
        self.max_ask = None

        self.traceback_window = None
        self.trade_off_score = None
        self.ask_score = None
        self.deploy_score = None
        self.last_score = None
        self.current_score = None
        self.wait_score = None
        self.reward = None
        self.current_step = None

        self.w = None
        self.decay = None
        self.tanh_scale = None
        self.deploy_quota_records = []
        self.deploy_parameter_records = []
        self.ask_quota_records = []

        if self.mode == 'run':
            self.seed(seed)
            self.reset()

            # we acturaly don't care about their borders because we only call back the shape.
            parameters_min = np.array([0, 0])
            parameters_max = np.array([1000, 1000])
            self.action_space = spaces.Tuple((spaces.Discrete(4),
                                              spaces.Box(parameters_min, parameters_max)))

            # we acturaly don't care about their borders because we only call back the shape.
            observation_min = np.array([0, 0, 0, 0, 0, 0, 0, 0])  # detailed see def get_states()
            observation_max = np.array([1, 1, 1, 1, 1, 1, 1, 1])
            self.observation_space = spaces.Box(observation_min, observation_max)

    def seed(self, seed: Optional[int] = None) -> list:
        self.np_random, seed = seeding.np_random(seed)  # noqa
        return [seed]

    def set_meta(self, info=None):
        self.max_step = info["max_step"]
        self.history = []  # info["history"]

        self.max_available_quota = info["max_available_quota"]
        self.min_deploy = info["min_deploy"]

        self.min_ask = info["min_ask"]
        self.max_ask = info["max_ask"]

        self.traceback_window = info["traceback_window"]
        self.trade_off_score = info["trade_off_score"]
        self.ask_score = info["ask_score"]
        self.deploy_score = info["deploy_score"]
        self.wait_score = info["wait_score"]
        self.last_score = info["last_score"]
        self.current_score = info["current_score"]
        self.current_step = info["current_step"]

        self.ask_quota_records = []
        self.deploy_parameter_records = []

    def set_weights(self, w=None, decay=None, tanh_scale=None):
        self.w = w
        self.decay = decay
        self.tanh_scale = tanh_scale

    def get_states(self):
        """
        only available when self.mode=='run'
        I think the weights of the scoring system should be also treated as states and return to the attacker
        self.traceback_window
        this is also a very important obersvations we need to know, you should also return self.traceback_window back to
        the agent
        """
        if self.mode == 'run':
            states = [
                self.max_available_quota,
                self.current_step / self.max_step,
                self.ask_score, # nan
                self.deploy_score,
                self.trade_off_score,
                self.wait_score,
                self.last_score,
                self.current_score
            ]
            return states
        else:
            raise ValueError("get_states() is not available when self.mode is '{}'".format(self.mode))

    def reset(self) -> list:
        """
        重置环境，将状态设置为初始状态，返回： 状态值
        :return:
        """
        if self.mode == 'run':
            # this is the traceback records to calculate some scores
            self.history = []
            # this is the traceback windows, we have len(self.history)<=self.traceback_window
            self.traceback_window = 10

            self.max_step = 40  # Normal(torch.tensor([100.0]), torch.tensor([30.0])).sample().item()  # .detach().cpu().numpy()

            self.max_available_quota = Normal(torch.tensor([20.0]), torch.tensor([10.0])).sample().item()

            self.ask_score = 0.0  # Uniform(torch.tensor([0.0]), torch.tensor([1.0])).sample().item()
            self.deploy_score = 0.0  # Uniform(torch.tensor([0.0]), torch.tensor([1.0])).sample().item()

            self.last_score = 0.0  # Uniform(torch.tensor([0.0]), torch.tensor([1.0])).sample().item()
            self.current_score = 0.0  # Uniform(torch.tensor([0.0]), torch.tensor([1.0])).sample().item()
            self.trade_off_score = 0.0
            self.wait_score = 0.0  # Uniform(torch.tensor([0.0]), torch.tensor([1.0])).sample().item()

            self.reward = 0
            self.current_step = 0
            self.min_ask, self.max_ask = 1, 23  # 实际情况中应该是根据真实数据分布采样得到
            self.min_deploy = 1
            self.deploy_quota_records = []
            self.ask_quota_records = []
            self.deploy_parameter_records = []

            return self.get_states()
        else:
            raise ValueError(
                "reset() is not available when self.mode is '{}'".format(self.mode))

    def get_details(self):
        """
        this function might be merged with get_states()
        """
        if self.mode == 'run':
            return {
                "max_step": self.max_step,
                "history": self.history,
                "max_available_quota": self.max_available_quota,
                "min_deploy": self.min_deploy,
                "min_ask": self.min_ask,
                "max_ask": self.max_ask,
                "trade_off_score": self.trade_off_score,
                "traceback_window": self.traceback_window,
                "ask_score": self.ask_score,
                "deploy_score": self.deploy_score,
                "wait_score": self.wait_score,
                "last_score": self.last_score,
                "current_score": self.current_score,
                "reward": self.reward,
                "current_step": self.current_step}
        else:
            raise ValueError("get_details() is not available when self.mode is '{}'".format(self.mode))

    def quota(self, par, min_quota, max_quota):

        """
        normal_parameter: actor value within (-1+-sigma,1+-sigma)
        return: a quota value within (min_ask_quota, max_ask_quota)
        """
        par = 0.5 * (par + 1)  # (-1+-sigma,1+-sigma)-->(0+-sigma,1+-sigma)
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

    def _decay_sum2(self, records, decay, condition_fun=None, exter_fun=None):
        """
        serve=:
        _update_ask_score
        _update_deploy_score
        _update_util_score
        _update_ask_fre_score
        _update_dep_fre_score
        _update_growth_score
        _update_wait_score
        """
        weighted_sum = 0.0
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

        return weighted_sum

    def _update_ask_score(self, ):

        weighted_ave_quota = self._decay_sum2(records=self.history,
                                              decay=self.decay[0],
                                              condition_fun=lambda x: x == 0,
                                              exter_fun=lambda x: x)


        self.ask_score = self.tanh(self.tanh_scale[0] * weighted_ave_quota)

    def _update_deploy_score(self, ):

        """

           26703
           00:740--->wait: 上期账单金额,上期还款金额:希望越少越好：因为这意味着用户不活跃了:action_type=3
           0x:426--->ask: 上期账单金额为0,上期还款金额:希望还款金额越多越好：因为这意味着用户有较强的还款能力: action_type=0
           x0:1785-->deploy：上期账单金额,上期还款金额为0:希望上期账单金额越少越好：因为这意味着用户有少的负债: action_type=1
           xx:23011-->trade-off:上期账单金额,上期还款金额:希望两者比值接近1：因为这意味着用户可以即时还清债务并且现金流充分: action_type=2

        """
        # we calculate weighted deploy quota with time decay and treat all ask action
        # accompanied by a copied deploy
        weighted_ave_quota = self._decay_sum2(records=self.history,
                                              decay=self.decay[1],
                                              condition_fun=lambda x: x == 1,
                                              exter_fun=lambda x: x)


        self.deploy_score = self.tanh(-self.tanh_scale[1] * weighted_ave_quota) + 1

    def _update_trade_off_score(self, ):
        """
                        26703
                        00:740--->wait: 上期账单金额,上期还款金额:希望越少越好：因为这意味着用户不活跃了:action_type=3
                        0x:426--->ask: 上期账单金额为0,上期还款金额:希望还款金额越多越好：因为这意味着用户有较强的还款能力: action_type=0
                        x0:1785-->deploy：上期账单金额,上期还款金额为0:希望上期账单金额越少越好：因为这意味着用户有少的负债: action_type=1
                        xx:23011-->trade-off:上期账单金额,上期还款金额:希望两者比值接近1：因为这意味着用户可以即时还清债务并且现金流充分: action_type=2
               """

        # we calculate weighted deploy quota with time decay and treat all ask action
        # accompanied by a copied deploy
        weighted_ave_quota = self._decay_sum2(records=self.history,
                                              decay=self.decay[2],
                                              condition_fun=lambda x: x == 2,
                                              exter_fun=lambda x: x[1] / x[0] - 1.0)



        self.trade_off_score = self.tanh(-self.tanh_scale[2] * weighted_ave_quota) + 1


    def _update_wait_score(self):

        """
                 26703
                 00:740--->wait: 上期账单金额,上期还款金额:希望越少越好：因为这意味着用户不活跃了:action_type=3
                 0x:426--->ask: 上期账单金额为0,上期还款金额:希望还款金额越多越好：因为这意味着用户有较强的还款能力: action_type=0
                 x0:1785-->deploy：上期账单金额,上期还款金额为0:希望上期账单金额越少越好：因为这意味着用户有少的负债: action_type=1
                 xx:23011-->trade-off:上期账单金额,上期还款金额:希望两者比值接近1：因为这意味着用户可以即时还清债务并且现金流充分: action_type=2
        """

        weighted_wait_num = self._decay_sum2(records=self.history, decay=self.decay[3],
                                             condition_fun=lambda x: x == 3, exter_fun=lambda q: 1)



        self.wait_score = self.tanh(-self.tanh_scale[3] * weighted_wait_num ) + 1

    def update_scores(self):

        """
           26703
           00:740--->wait: 上期账单金额,上期还款金额:希望越少越好：因为这意味着用户不活跃了:action_type=3
           0x:426--->ask: 上期账单金额为0,上期还款金额:希望还款金额越多越好：因为这意味着用户有较强的还款能力: action_type=0
           x0:1785-->deploy：上期账单金额,上期还款金额为0:希望上期账单金额越少越好：因为这意味着用户有少的负债: action_type=1
           xx:23011-->trade-off:上期账单金额,上期还款金额:希望两者比值接近1：因为这意味着用户可以即时还清债务并且现金流充分: action_type=2
        """
        self._update_wait_score()
        self._update_ask_score()
        self._update_deploy_score()
        self._update_trade_off_score()


        self.current_score = self.w[0] * self.ask_score + self.w[1] * self.deploy_score + \
                             self.w[2] * self.trade_off_score + self.w[3] * self.wait_score

        self.current_score = self.tanh(self.tanh_scale[4] * self.current_score)

    def get_reward(self, action):

        """
                  26703
                  00:740--->wait: 上期账单金额,上期还款金额:希望越少越好：因为这意味着用户不活跃了:action_type=3
                  0x:426--->ask: 上期账单金额为0,上期还款金额:希望还款金额越多越好：因为这意味着用户有较强的还款能力: action_type=0
                  x0:1785-->deploy：上期账单金额,上期还款金额为0:希望上期账单金额越少越好：因为这意味着用户有少的负债: action_type=1
                  xx:23011-->trade-off:上期账单金额,上期还款金额:希望两者比值接近1：因为这意味着用户可以即时还清债务并且现金流充分: action_type=2
        """

        if self.mode == 'test':
            r = torch.tensor(0.0, requires_grad=True)
        else:
            r = 0

        branch = 0

        delta_s = self.tanh(self.abs(self.current_score - self.last_score)) * 10  # 10

        if action.id == 0:
            """
            0x:426--->ask: 上期账单金额为0,上期还款金额:希望还款金额越多越好：因为这意味着用户有较强的还款能力: action_type=0
            """
            quota = self.quota(action.parameter, min_quota=self.min_ask, max_quota=self.max_ask)
            if self.current_score > self.last_score:
                r = -quota * delta_s
                branch = 1
            else:
                r = quota * delta_s
                branch = 2
        elif action.id == 1:  # 普通难度A
            """
            x0:1785-->deploy：上期账单金额,上期还款金额为0:希望上期账单金额越少越好：因为这意味着用户有少的负债: action_type=1
            """
            quota = self.quota(action.parameter, min_quota=self.min_deploy, max_quota=self.max_available_quota)
            if self.current_score > self.last_score:
                r = quota * delta_s
                branch = 3
            else:
                r = -quota * delta_s
                branch = 4
        elif action.id == 2:
            """
            xx:23011-->trade-off:上期账单金额,上期还款金额:希望两者比值接近1：因为这意味着用户可以即时还清债务并且现金流充分: action_type=2
            """
            [par0, par1] = action.parameter
            # print("xixxi",self.current_score,action.parameter)
            if self.current_score > self.last_score:
                r = self.abs(par0 / par1 - 1) * delta_s
                branch = 5
            else:
                r = -self.abs(par0 / par1 - 1) * delta_s
                branch = 6
        elif action.id == 3:
            """
            00:740--->wait: 上期账单金额,上期还款金额:希望越少越好：因为这意味着用户不活跃了:action_type=3
            """
            if self.current_score > self.last_score:
                r = 0.001
                branch = 7
            else:
                r = 0
                branch = 8

        return r, branch

    def step(self, raw_action: Tuple[int, list], init_step_flag=False) -> Tuple[list, float, bool, dict]:
        """
        注意：这里的action parameter是(-1+-sigma,1+-sigma)的数值，你需要对其进行scale后才能变成具体的quota，当然也许在后期的reward中，你也许并不需要绝对的quota数量



        """
        action = Action(*raw_action)
        # print("action.parameters: ", action.parameters)

        """
            26703
            00:740--->wait: 上期账单金额,上期还款金额:希望越少越好：因为这意味着用户不活跃了:action_type=3
            0x:426--->ask: 上期账单金额为0,上期还款金额:希望还款金额越多越好：因为这意味着用户有较强的还款能力: action_type=0
            x0:1785-->deploy：上期账单金额,上期还款金额为0:希望上期账单金额越少越好：因为这意味着用户有少的负债: action_type=1
            xx:23011-->trade-off:上期账单金额,上期还款金额:希望两者比值接近1：因为这意味着用户可以即时还清债务并且现金流充分: action_type=2
        """
        deploy_quota = 0.0
        ask_quota = 0.0
        deploy_parameter = 0.0
        if action.id == 0:
            ask_quota = self.quota(action.parameter, min_quota=self.min_ask, max_quota=self.max_ask)
            deploy_quota = 0
        elif action.id == 1:
            ask_quota = 0
            deploy_quota = self.quota(action.parameter, min_quota=self.min_deploy, max_quota=self.max_available_quota)
            deploy_parameter = action.parameter
        elif action.id == 2:
            [par0,par1]=action.parameter
            ask_quota = self.quota(par0, min_quota=self.min_ask, max_quota=self.max_ask)
            deploy_quota = self.quota(par1, min_quota=self.min_deploy, max_quota=self.max_available_quota)
        elif action.id == 3:
            ask_quota = 0
            deploy_quota = 0

        self.ask_quota_records.append(ask_quota)
        self.deploy_quota_records.append(deploy_quota)
        self.deploy_parameter_records.append(deploy_parameter)

        self.history.append(action)

        if len(self.history) > self.traceback_window:
            self.history = self.history[len(self.history) - self.traceback_window:]
            self.ask_quota_records = self.ask_quota_records[len(self.ask_quota_records) - self.traceback_window:]
            self.deploy_quota_records = self.deploy_quota_records[
                                        len(self.deploy_quota_records) - self.traceback_window:]
            self.deploy_parameter_records = self.deploy_parameter_records[
                                            len(self.deploy_parameter_records) - self.traceback_window:]

        self.current_step += 1

        self.update_scores()

        if self.mode == 'run':
            if self.current_step >= self.max_step:  # or all entries in self.history are action.id=="wait"
                done = True
            else:
                done = False
            self.reward, branch = self.get_reward(action)

            states = self.get_states()
            self.last_score = self.current_score
            return states, self.reward, done, {'branch': branch,
                                               'deploy_quota': deploy_quota,
                                               'render': {
                                                   'weight_factor_score': [self.w[0] * self.ask_score,
                                                                           self.w[1] * self.deploy_score,
                                                                           self.w[2] * self.trade_off_score,
                                                                           self.w[3] * self.wait_score
                                                                           ]}
                                               }

        elif self.mode == 'test':
            self.reward, branch = self.get_reward(action)
            self.last_score = self.current_score
            return self.reward


if __name__ == '__main__':
    import gym_hybrid

    env = gym.make('CreditEnv-v0')
    # env = gym.make("CartPole-v0")
    init_state = env.reset()
    N_F = env.observation_space.shape[0]
    N_A = 2  # env.action_space.n

    states = []
    actions = []
    rewards = []
    is_done = False
    final_r = 0
    state = init_state

    history = [(0, 1000), (1, 100), (1, 1000), (1, 100), (0, 100), (1, 200)]

    for act in history:
        # (action_id,quota)
        action = (act[0], [act[1]])  # parameter=tensor([[113.3151]], grad_fn=<ClampBackward1>), or 0
        next_state, reward, done, _ = env.step(action)  # raw_action: Tuple[int, list]
        print("state: ", next_state)
        print("reward: ", reward)
        """
            self.max_available_quota,
            self.current_step / self.max_step,
            self.ask_fre_score,
            self.dep_fre_score,
            self.ask_score,
            self.deploy_score,
            self.uti_score,
            self.last_score,
            self.current_score
        """
