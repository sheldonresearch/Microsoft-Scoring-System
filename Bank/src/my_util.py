import os
import torch
from models import PPO
from gym_hybrid.environments_beta import CreditEnv


def generate_corpus(attack_version=0,
                    credit_version=0,
                    credit_parameters=None,
                    max_ep_len=50,
                    total_test_episodes=50):
    credit = credit_parameters[credit_version]
    weight = credit['weight']
    decay = credit['decay']
    tanh_scale = credit['tanh_scale']
    w = tuple(torch.squeeze(weight).detach().numpy())
    de = tuple(torch.squeeze(decay).detach().numpy())
    ts = tuple(torch.squeeze(tanh_scale).detach().numpy())

    max_ep_len = max_ep_len
    total_test_episodes = total_test_episodes  # total num of testing episodes
    env = CreditEnv(mode='run')
    env.set_weights(w, de, ts)
    # state space dimension
    state_dim = env.observation_space.shape[0]
    # action space dimension
    discrete_action_space = env.action_space[0]
    continuous_action_space = env.action_space[1]
    discrete_action_dim = discrete_action_space.n
    continuous_action_dim = continuous_action_space.shape[0]
    action_dim = [discrete_action_dim, continuous_action_dim]
    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim)
    directory = "PPO_preTrained" + '/'
    checkpoint_path = directory + "PPO_CreditEnv-v{}.pth".format(attack_version)
    print("loading network from : " + checkpoint_path)
    ppo_agent.load(checkpoint_path)

    test_trajectories = []
    env_details = []
    accumulated_rewards = []
    accumulated_branches=[]

    ask_quota_records=[]
    deploy_quota_records=[]
    render=[]

    for ep in range(1, total_test_episodes + 1):
        ep_rewards = []
        ep_branches=[]
        ep_ask_quota_records=[]
        ep_deploy_quota_records=[]
        ep_render=[]


        state = env.reset()
        env_details.append(env.get_details().copy())

        for t in range(1, max_ep_len + 1):
            action = ppo_agent.select_action(state, init_step_flag=False)
            # print(action)
            state, reward, done, infor = env.step(action,init_step_flag=False)
            ep_rewards.append(reward)
            ep_branches.append(infor['branch'])
            ep_ask_quota_records.append(env.ask_quota_records[-1])
            ep_deploy_quota_records.append(env.deploy_quota_records[-1])
            ep_render.append(infor['render'])
            # if done:
            #     break
        print(
            "attacker_v {} credit_v {} episode {}/{} ".format(attack_version, credit_version, ep, total_test_episodes),
            ppo_agent.buffer.actions)

        # print(
        #     "attacker_v {} credit_v {} episode {}/{} ".format(attack_version, credit_version, ep, total_test_episodes),
        #     ep_rewards)

        accumulated_rewards.append(ep_rewards)
        accumulated_branches.append(ep_branches)
        test_trajectories.append(ppo_agent.buffer.actions.copy())
        ask_quota_records.append(ep_ask_quota_records)
        deploy_quota_records.append(ep_deploy_quota_records)
        render.append(ep_render)

        ppo_agent.buffer.clear()
    env.close()

    return test_trajectories, env_details, accumulated_rewards,accumulated_branches,ask_quota_records,deploy_quota_records,render


def attacker_train(version=0,
                   w=(1, 1, 1, 1, 1),
                   decay=(0.05, 0.05, 0.05, 0.05, 0.05, 1),
                   tanh_scale=(0.0001, 0.0001, 0.01, 0.0001, 0.0001, 0.001, 1), config=None):
    """
    acturally, each attacker should be trained via different seeds and then we save all these
    this function is relocated in my_util.py instead of original gradient_booster_attacker.py
    because this function might be also reused in train_attacker_alone.py
    """

    env_name = "CreditEnv-v{}".format(version)
    max_ep_len = config.max_ep_len
    max_training_timesteps = config.max_training_timesteps
    print_freq = config.print_freq
    log_freq = config.log_freq
    save_model_freq = config.save_model_freq
    action_std = config.action_std
    action_std_decay_rate = config.action_std_decay_rate
    min_action_std = config.min_action_std
    action_std_decay_freq = config.action_std_decay_freq

    update_timestep = config.update_timestep
    K_epochs = config.K_epochs

    eps_clip = config.eps_clip
    gamma = config.gamma

    lr_actor = config.lr_actor
    lr_critic = config.lr_critic

    env = CreditEnv(mode='run')
    env.set_weights(w, decay, tanh_scale)


    state_dim = env.observation_space.shape[0]  # 9
    discrete_action_space = env.action_space[0]
    continuous_action_space = env.action_space[1]
    discrete_action_dim = discrete_action_space.n
    continuous_action_dim = continuous_action_space.shape[0]
    action_dim = [discrete_action_dim, continuous_action_dim]

    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_f_name = log_dir + '/PPO_' + env_name + "_log.csv"

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}.pth".format(env_name)

    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)


    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0
    print("attacker run...")
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len + 1):
            # select action with policy
            action = ppo_agent.select_action(state,init_step_flag=False)
            state, reward, done, _ = env.step(action,init_step_flag=False)#init_step_flag)
            # print(action)



            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            if time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)
            # log in logging file
            if time_step % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                # log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                # print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                        print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                ppo_agent.save(checkpoint_path)

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()



# def real_data_test():
#     env_name = "CreditEnv-v0"
#     raise NotImplementedError('this function need further polish')
#     max_ep_len = 400
#     action_std = 0.1
#     total_test_episodes = 10  # total num of testing episodes
#     K_epochs = 80  # update policy for K epochs
#     eps_clip = 0.2  # clip parameter for PPO
#     gamma = 0.99  # discount factor
#     lr_actor = 0.0003  # learning rate for actor
#     lr_critic = 0.001  # learning rate for critic
#     env = CreditEnv(mode='test')
#     # state space dimension
#     state_dim = env.observation_space.shape[0]
#
#     # action space dimension
#     discrete_action_space = env.action_space[0]
#     continuous_action_space = env.action_space[1]
#     discrete_action_dim = discrete_action_space.n
#     continuous_action_dim = continuous_action_space.shape[0]
#     action_dim = [discrete_action_dim, continuous_action_dim]
#     print("action_dim: ", action_dim)
#
#     # initialize a PPO agent
#     ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)
#
#     # preTrained weights directory
#     random_seed = 0  #### set this to load a particular checkpoint trained on random seed
#     run_num_pretrained = 2  #### set this to load a particular checkpoint num
#
#     directory = "PPO_preTrained" + '/' + env_name + '/'
#     checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
#     print("loading network from : " + checkpoint_path)
#
#     ppo_agent.load(checkpoint_path)
#
#     print("--------------------------------------------------------------------------------------------")
#
#     test_running_reward = 0
#
#     for ep in range(1, total_test_episodes + 1):
#         ep_reward = 0
#         state = env.reset()
#         states = []
#         rewards = []
#         for t in range(1, max_ep_len + 1):
#             action = ppo_agent.select_action(state)
#             state, reward, done, _ = env.step(action)
#             states.append(state)
#             rewards.append(reward)
#             ep_reward += reward
#             if done:
#                 break
#
#         # clear buffer
#         print(ppo_agent.buffer.actions)
#         print(states)
#         print(rewards)
#         ppo_agent.buffer.clear()
#
#         test_running_reward += ep_reward
#         print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
#         ep_reward = 0
#
#     env.close()
#
#     print("============================================================================================")
#
#     avg_test_reward = test_running_reward / total_test_episodes
#     avg_test_reward = round(avg_test_reward, 2)
#     print("average test reward : " + str(avg_test_reward))
#
#     print("============================================================================================")
#

if __name__ == '__main__':
    pass
