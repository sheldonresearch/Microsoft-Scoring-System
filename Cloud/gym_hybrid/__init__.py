from gym.envs.registration import register
from gym_hybrid.environments_beta import CreditEnv

register(
    id='CreditEnv-v0',
    entry_point='gym_hybrid:CreditEnv',
)