import torch
from torch.nn.functional import softmax
import pickle
from config import Config
from my_util import generate_corpus, attacker_train
from models import GradientBooster
import copy

if __name__ == '__main__':

    c = Config().gba

    cre_par = c.ini_credit_parameters
    weight = cre_par["weight"]
    decay = cre_par["decay"]
    tanh_scale = cre_par["tanh_scale"]
    cre_par = {
        'for_v': 0,
        'weight': copy.deepcopy(weight.detach()),
        'decay': copy.deepcopy(decay.detach()),
        'tanh_scale': copy.deepcopy(tanh_scale.detach())
    }

    credit_parameters = [cre_par]
    booster_training_loss = []
    save_parameter_loss_fre = 1

    for v in range(c.max_version):
        w = tuple(torch.squeeze(weight).detach().numpy())
        de = tuple(torch.squeeze(decay).detach().numpy())
        ts = tuple(torch.squeeze(tanh_scale).detach().numpy())

        attacker_train(version=v, w=w, decay=de, tanh_scale=ts, config=c)

        test_trajectories, env_details, accumulated_rewards, _, _, _,_ = generate_corpus(attack_version=v,
                                                                                       credit_version=v,
                                                                                       credit_parameters=credit_parameters,
                                                                                       max_ep_len=c.test_max_ep_len,
                                                                                       total_test_episodes=c.total_test_episodes)

        gb = GradientBooster(init_weight=weight, init_decay=decay, init_tanh_scale=tanh_scale, lr=c.booster_lr)

        loss_record = []
        for k in range(c.booster_epoch):
            gb.optimizer.zero_grad()
            loss= gb.reward_loss(test_trajectories=test_trajectories, env_details=env_details)
            loss_record.append(loss.item())
            loss.backward()
            gb.optimizer.step()

        booster_training_loss.append(loss_record)

        weight = gb.weight
        weight = softmax(weight, dim=0)
        decay = torch.clip(gb.decay, min=0.0001, max=1.0)
        tanh_scale = torch.clip(gb.tanh_scale, min=0.0001, max=10)

        cre_par = {
            'for_v': v + 1,
            'weight': copy.deepcopy(weight.detach()),
            'decay': copy.deepcopy(decay.detach()),
            'tanh_scale': copy.deepcopy(tanh_scale.detach())
        }

        credit_parameters.append(cre_par)

        del gb

        if v % save_parameter_loss_fre == 0:
            print("saving credit parameters from version 0 to version{}...".format(v))
            pickle.dump(credit_parameters, open('Var/credit_parameters.list.dic', 'wb'))
            print(credit_parameters)

            print("saving booster training loss from version 0 to version{}...".format(v))
            pickle.dump(booster_training_loss, open('Var/booster_training_loss.list', 'wb'))

        print("all done!")

    pickle.dump(credit_parameters, open('Var/credit_parameters.list.dic', 'wb'))
