import torch
import torch.nn as nn
# from distance.soft_dtw import SoftDTW
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle


def _decay_sum2(decay):
    window_size = 30
    decay_items = []
    for ii in range(window_size):
        t = window_size - ii - 1
        dec = np.exp(-decay * t)
        decay_items.append(dec)

    return decay_items


credits = pickle.load(open('src/Var/credit_parameters.list.dic', 'rb'))
print(credits[18])
decay = credits[18]['decay']
weight=credits[18]['weight']
print(decay.detach().numpy())

# plt.figure(figsize=(7, 3))

# weight change
sns.heatmap(list(np.exp(weight.detach().numpy())),
            fmt='d', linewidths=.0, cmap='YlGnBu', cbar=True)  # ,vmin=0.2,vmax=1.0)

# #decay change
# sns.heatmap([_decay_sum2(0.05),
#              _decay_sum2(decay[0, 0]),
#              _decay_sum2(decay[1, 0]),
#              _decay_sum2(decay[2, 0]),
#              _decay_sum2(decay[3, 0]),
#              _decay_sum2(decay[4, 0]),
#              _decay_sum2(decay[5, 0])],
#             fmt='d', linewidths=.0, cmap='YlGnBu', cbar=True,
#             yticklabels=['init', 'ask', 'dep', 'uti', 'ask_fre', 'dep_fre', 'grow'])  # ,vmin=0.2,vmax=1.0)
plt.tight_layout()
# plt.subplot(252)
# sns.heatmap( [_decay_sum2(0.0001)], fmt='d', linewidths=.5, cmap='YlGnBu',yticklabels=['deploy core decay'])
#


plt.show()

# dat = pd.read_excel('behavior.xlsx')
# data = dat[dat['uid'] == 'd047ded8-3184-4052-94d4-18368976ce98']
# data = data.sort_values(by='action_index', ascending=True)
# data = data.reset_index(drop=True)
# print(data)
#
# deploy_list = []
# remain_quota=[]
# x = 0
# for _, row in data.iterrows():
#     max_available_quota=row['max_available_quota']
#     uid = row['uid']
#     atype = row['action_type']
#     apara = row['action_quota']
#     x = row['action_index']
#     remain_quota.append([x, apara, 'ask'])
#     if atype == 1:
#         deploy_list.append([x, apara, 'deploy'])
#         remain_quota.append([x,max_available_quota-apara])
#     elif atype == 0:
#         print("ask quota: ",apara)
#         deploy_list.append([x, apara, 'ask'])
#         remain_quota.append(remain_quota[-1])
#     else:
#         deploy_list.append([x, -10, 'wait'])
#         remain_quota.append([x, max_available_quota])
#
# # data = pd.DataFrame(data=remain_quota, columns=['step', 'quota', 'type'])
# data = pd.DataFrame(data=deploy_list, columns=['step', 'quota', 'type'])
#
# plt.figure(figsize=(7, 3))
# pal = {'deploy': 'green', 'ask': 'red', 'wait': 'yellow'}
# ax = sns.barplot(x="step", y="quota", data=data, hue='type', palette=pal)  # sns.color_palette("hls", 3))
# ax.set_xticks(ticks=[i for i in range(-1, 100, 10)])
# # action reward (scaled to 0-1)
# # plt.plot([0.5 for i in range(100)])
#
# # acumulative reward (scale to 0-1)
# plt.show()

# A=41114.82
# B=3.3869
# C=3.0609
# D=29000
#
# a=(A*B+D)*1.0/(A+D*1.0/C)
# print(a)
#
#
#
# """
# ==============================================
# discrete_action_logprobs
#  torch.Size([1000]) 100*10
#  tensor([-1.1970, -1.1692, -1.1746, -0.9525, -1.1819, -1.1731, ...
#
# continuous_action_logprobs
#  torch.Size([1000])
#  tensor([-2.1886, -2.5237, ...
#
# state_values
#  torch.Size([1000, 1])
#  tensor([[-0.1194],,...
#
# discrete_dist_entropy
#  torch.Size([1000])
#  tensor([1.0938, 1.0931, ...
#
# continuous_dist_entropy
#  torch.Size([1000])
#  tensor([2.7243, 2.7243, ...
#
# discrete_dist_probs
#  torch.Size([1000, 3])
#  tensor([[0.3021, 0.3188, 0.3791],
#         [0.3063, 0.3106, 0.3831],
#         [0.3060, 0.3089, 0.3850],
#         ...,
#         [0.3553, 0.2905, 0.3542],
#         [0.3556, 0.2904, 0.3539],
#         [0.3557, 0.2904, 0.3540]], grad_fn=<DivBackward0>)
#
#
# """
#
#
# # self.corpus = np.random.choice(self.corpus, size=int(len(self.corpus) * 0.01))
#
#
# def get_dist_by_quota(self, corpus, actions, sample=1.0):
#     """
#     this function is used in attacker loss
#
#     Method 1：
#     初步构想：距离包含两个项目（1）离散模式（2）quota序列
#
#     离散模式：根据新的discrete_dist_probs，计算其与真实数据的one_hot_discrete_actions的距离
#
#     quota序列：
#     - 根据过去数据的动作类型重新计算对应的para：得到新的continuous_action (action.parameter)
#     - 在buffer中记录每个动作对应的min_ask,max_ask,min_dep,max_dep
#     - 根据buffer中的min,max，重新构建出新的quota序列
#     - 计算这个quota序列和真实序列的相似性
#
#     Method 2：
#     仅仅计算离散模式
#
#
#     Method 3：
#     仅仅计算宏观统计量
#
#
#     Method 4[recommend!]：
#     离散模式和Method1相同，
#     quota序列这里，想办法把真实数据的quota序列映射到parameters中：
#     对于一条序列：
#     找到min ask quota, max ask quota, min deploy quota, 各个时间点的max available quota（数据集天然携带）
#     当前如果是ask q, parameter=(q-min_ask)/(max-min)*2-1
#     当前如果是dep q, parameter=(q-min_dep)/(max available quota at t-min deploy)*2-1
#
#     """
#     # actions=torch.unsqueeze(actions,dim=1)
#     criterion = SoftDTW(gamma=1.0, normalize=True)
#     dis_loss = []
#     """
#       [
#         [[1,0], [0,1],[0,1],[0,1],[0,1],[0,1],[0,1]],
#         [[1,0], [1,0], [0,1],[0,1],[0,1],[0,1], [1,0], [0,1]],
#         [[1,0], [0,1], [1,0],[0,1],[0,1],[0,1]]
#       ]
#     """
#     # corpus=np.random.choice(corpus,size=int(len(corpus)*sample))
#
#     for cor in corpus:
#         y = torch.tensor(cor, dtype=torch.float32).detach()
#         print("actions:",actions)
#         print("y: ",y)
#         loss = criterion(actions, y)
#         dis_loss.append(loss)
#
#     dis_loss = torch.stack(dis_loss, dim=0)
#     dis_loss_mean = dis_loss.mean()
#     # print("dis_loss_mean: ",dis_loss_mean)
#     # print("tanh dis_loss_mean: ",tanh(dis_loss_mean/500))
#     return torch.tanh((dis_loss_mean / 500))  # dis_loss.mean()
#
#
# def get_dist_by_score():
#     """
#     this functin is used in booster loss
#
#     """
