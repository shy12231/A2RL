# -*- coding: utf-8 -*-
import os
import math
import argparse
import random
import numpy
import torch
import torch.nn as nn
from bucket_iterator import BucketIterator
from sklearn import metrics
from data_utils import ABSADatesetReader
from models import LSTM, ASCNN, ASGCN, ASTCN
from models import a2rl as rl

import datetime

class Instructor:
    def __init__(self, opt):
        self.opt = opt

        absa_dataset = ABSADatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim)
        self.absa_dataset = absa_dataset
        
        self.train_data_loader = BucketIterator(data=absa_dataset.train_data, batch_size=opt.batch_size, shuffle=True)
        self.test_data_loader = BucketIterator(data=absa_dataset.test_data, batch_size=opt.batch_size, shuffle=True)
        
        self.actorModel = rl.actor(self.opt).to(opt.device)
        self.criticModel = rl.critic(absa_dataset.embedding_matrix, self.opt).to(opt.device)
        
        self._print_args()
        self.global_f1_pre = 0.
        self.global_f1 = 0.

        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=opt.device.index))

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.actorModel.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params

        for p in self.criticModel.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params

        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for p in self.criticModel.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

        for p in self.actorModel.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, actorModel, criticModel, RL_train, LSTM_train):
        max_test_acc_pre = 0
        max_test_f1_pre = 0

        max_test_acc = 0
        max_test_f1 = 0

        global_step = 0
        continue_not_increase = 0

        # critic定义两个相同的网络, target和active, target更新缓慢,用于表示真实值 active更新多,用于表示预测值
        critic_target_optimizer = torch.optim.Adam(criticModel.target_pred.parameters())
        critic_active_optimizer = torch.optim.Adam(criticModel.active_pred.parameters())

        # actor定义两个相同的网络, target和active, target更新缓慢,用于表示真实值 active更新多,用于表示预测值
        actor_target_optimizer = torch.optim.Adam(actorModel.target_policy.parameters())
        actor_active_optimizer = torch.optim.Adam(actorModel.active_policy.parameters())

        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            n_correct, n_total = 0, 0
            increase_flag = False
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1
                totloss = 0.
                # 每个batch将target网络的值复制到active网络 保持两者一致
                criticModel.assign_active_network()
                actorModel.assign_active_network()
                aveloss = 0.
                # 原始数据
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)

                if RL_train or LSTM_train:
                    # 从batch中取  文本下标, 方面下标, 左文本下标, 邻接矩阵
                    text_indices, aspect_indices, left_indices, adjacent = inputs
                    aspect_len = torch.sum(aspect_indices != 0, dim=-1) # 方面词的长度
                    left_len = torch.sum(left_indices != 0, dim=-1) # 方面词之前的句子长度
                    aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len + aspect_len-1).unsqueeze(1)], dim=1) # 方面词开始和结束的下标
                    targets = torch.autograd.Variable(targets).long()
                    pred = torch.zeros(len(text_indices), 3).cuda()

                    for sent in range(len(text_indices)):
                        text_indices_sent = text_indices[sent].unsqueeze(0)
                        text_len = torch.sum(text_indices_sent != 0, dim=1)

                        targets_sent = targets[sent].unsqueeze(0)
                        aspect_indices_sent = aspect_indices[sent].unsqueeze(0)
                        left_indices_sent = left_indices[sent].unsqueeze(0)
                        aspect_double_idx_sent = aspect_double_idx[sent]
                        adjacent_sent = adjacent[sent][:text_len,:text_len]
                        text_indices_sent_rl = text_indices[sent][:text_len].unsqueeze(0)
                        if RL_train:
                            # actor 训练
                            criticModel.train(False)
                            actorModel.train()
                            actionlist, statelist, losslist = [], [], []
                            aveLoss = 0.
                            for i in range(self.opt.samplecnt):
                                # 通过采样, 得到文本的动作序列,状态和采样后上下文表示 ation torch.Size(length, 1)
                                text_vec_sent = criticModel.wordvector_find(text_indices_sent_rl, aspect_double_idx_sent.unsqueeze(0), text_len, aspect_len[sent].unsqueeze(0))
                                actions, states, Rinput, Rlength, aspect_last = rl.Sampling_RL(actorModel, criticModel, text_indices_sent, text_vec_sent, aspect_double_idx_sent, self.opt.epsilon)

                                actionlist.append(actions)
                                statelist.append(states)

                                adj = adjacent_sent[:].unsqueeze(0)
                                actions = torch.tensor(actions).cuda()
                                input_critic = [text_indices_sent_rl, aspect_indices_sent, left_indices_sent, adj, actions]
                                # 对采样后的上下文表示进行分类, 分类后的损失作为reward
                                with torch.no_grad():
                                    predicted, _ = criticModel(input_critic, scope = "target")

                                loss_cross = criterion(predicted, targets_sent)
                                loss_punish = (float(Rlength) / text_len[0].cpu().numpy()) ** 2 * self.opt.loss_punish_ratio
                                loss_ = loss_cross + loss_punish

                                aveloss += loss_
                                losslist.append(loss_)

                            prediction, _ = criticModel(input_critic, scope = "target")
                            pred[sent] = prediction

                            aveloss /= self.opt.samplecnt
                            totloss += aveloss
                            grad1 = None
                            grad2 = None
                            grad3 = None
                            grad4 = None
                            flag = 0
                        
                            # actor根据公式 更新
                            for i in range(self.opt.samplecnt):
                                for pos in range(len(actionlist[i])):
                                    rr = [0, 0]
                                    rr[actionlist[i][pos]] = ((losslist[i] - aveloss) * self.opt.alpha).cpu().item()
                                    g = actorModel.get_gradient(statelist[i][pos][0], statelist[i][pos][1], statelist[i][pos][2], rr, scope = "target")
                                    if flag == 0:
                                        grad1 = g[0]
                                        grad2 = g[1]
                                        grad3 = g[2]
                                        grad4 = g[3]
                                        flag = 1
                                    else:
                                        grad1 += g[0]
                                        grad2 += g[1]
                                        grad3 += g[2]
                                        grad4 += g[3]
                            actor_target_optimizer.zero_grad()
                            actor_active_optimizer.zero_grad()
                            
                            actorModel.assign_active_network_gradients(grad1, grad2, grad3, grad4)
                            actor_active_optimizer.step()

                        if LSTM_train:
                            # critic 训练
                            criticModel.train()
                            actorModel.train(False)

                            text_vec_sent = criticModel.wordvector_find(text_indices_sent_rl, aspect_double_idx_sent.unsqueeze(0), text_len, aspect_len[sent].unsqueeze(0))
                            actions, states, Rinput, Rlength, aspect_last = rl.Sampling_RL(actorModel, criticModel, text_indices_sent, text_vec_sent, aspect_double_idx_sent, Random=False)

                            adj = adjacent_sent[:].unsqueeze(0)
                            actions = torch.tensor(actions).cuda()

                            input_critic = [text_indices_sent_rl, aspect_indices_sent, left_indices_sent, adj, actions]

                            critic_active_optimizer.zero_grad()
                            critic_target_optimizer.zero_grad()

                            prediction, _ = criticModel(input_critic, scope = "target")
                            pred[sent] = prediction
                            loss = criterion(prediction, targets_sent)
                            loss.backward()
                            totloss += loss

                            criticModel.assign_active_network_gradients()
                            critic_active_optimizer.step()
                else:
                    criticModel.train()
                    actorModel.train(False)
                    critic_active_optimizer.zero_grad()
                    critic_target_optimizer.zero_grad()
                    
                    text_indices, aspect_indices, left_indices, adjacent = inputs
                    actions = None
                    inputs = [text_indices, aspect_indices, left_indices, adjacent, actions]
                    
                    pred, _ = criticModel(inputs, scope = "target")
                    totloss = criterion(pred, targets)
                    totloss.backward()
                    criticModel.assign_active_network_gradients()
                    critic_active_optimizer.step()

                if RL_train or LSTM_train:
                    if RL_train:
                        actorModel.train()
                        criticModel.train(False)
                        actorModel.update_target_network(self.opt.tau)
                    if LSTM_train:
                        actorModel.train(False)
                        criticModel.train()
                        criticModel.update_target_network(self.opt.tau)
                else:
                    criticModel.train()
                    actorModel.train(False)
                    criticModel.assign_target_network()

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(pred, -1) == targets).sum().item()
                    n_total += len(pred)
                    train_acc = n_correct / n_total

                    if RL_train or LSTM_train:
                        test_acc, test_f1 = self._evaluate_acc_f1_rl(actorModel, criticModel)
                        if test_acc > max_test_acc:
                            max_test_acc = test_acc
                        if test_f1 > max_test_f1:
                            increase_flag = True
                            max_test_f1 = test_f1

                            if self.opt.save and test_f1 > self.global_f1:
                                self.global_f1 = test_f1
                                torch.save(criticModel.state_dict(), 'A2RL/state_dict_temp/'+self.opt.dataset+'_'+str(self.opt.loss_punish_ratio)+'_critic.pkl')
                                torch.save(actorModel.state_dict(), 'A2RL/state_dict_temp/'+self.opt.dataset+'_'+str(self.opt.loss_punish_ratio)+'_actor.pkl')
                                print('>>> best model saved.')

                            if self.opt.save and self.global_f1 > self.global_f1_all:
                                self.global_f1_all = self.global_f1
                                torch.save(criticModel.state_dict(), 'A2RL/state_dict/'+self.opt.dataset+'_'+str(self.opt.loss_punish_ratio)+'_critic_all.pkl')
                                torch.save(actorModel.state_dict(), 'A2RL/state_dict/'+self.opt.dataset+'_'+str(self.opt.loss_punish_ratio)+'_actor_all.pkl')
                                print('>>> best all saved.')
                        print('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, test_f1: {:.4f}'.format(totloss.item(), train_acc, test_acc, test_f1))
                    else:
                        # critic 热启动
                        test_acc, test_f1 = self._evaluate_acc_f1(criticModel)
                        if test_acc > max_test_acc_pre:
                            max_test_acc_pre = test_acc
                        if test_f1 > max_test_f1_pre:
                            increase_flag = True
                            max_test_f1_pre = test_f1
                            if self.opt.save and test_f1 > self.global_f1_pre:
                                self.global_f1_pre = test_f1
                                torch.save(criticModel.state_dict(), 'A2RL/state_dict_temp/'+self.opt.dataset+'_'+str(self.opt.loss_punish_ratio)+'_pre.pkl')
                                print('>>> best pre model saved.')
                        print('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, test_f1: {:.4f}'.format(totloss.item(), train_acc, test_acc, test_f1))
                        max_test_acc, max_test_f1 = max_test_acc_pre, max_test_f1_pre
            # 超过5次不增加早停
            if increase_flag == False:
                continue_not_increase += 1
                if continue_not_increase >= self.opt.early_stop + (int(RL_train==False) * int(LSTM_train==False) * 3):
                    print('early stop.')
                    break
            else:
                continue_not_increase = 0
        return max_test_acc, max_test_f1

    def _evaluate_acc_f1(self, criticModel):
        # switch model to evaluation mode
        self.criticModel.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs = [t_sample_batched[col].to(opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(opt.device)
                
                text_indices, aspect_indices, left_indices, adjacent = t_inputs
                actions = None
                t_inputs = text_indices, aspect_indices, left_indices, adjacent, actions
                
                t_outputs, _ = self.criticModel(t_inputs, scope = "target")

                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return test_acc, f1

    def _evaluate_acc_f1_rl(self, actorModel, criticModel):
        # switch model to evaluation mode
        actorModel.eval()
        criticModel.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, pred_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                
                t_inputs = [t_sample_batched[col].to(opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(opt.device)
                
                text_indices, aspect_indices, left_indices, adjacent = t_inputs
                aspect_len = torch.sum(aspect_indices != 0, dim=-1)
                left_len = torch.sum(left_indices != 0, dim=-1)
                aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
                pred = torch.zeros(len(text_indices), 3).cuda()

                for sent in range(len(text_indices)):
                    text_indices_sent = text_indices[sent].unsqueeze(0)
                    text_len = torch.sum(text_indices_sent != 0, dim=1)

                    aspect_indices_sent = aspect_indices[sent].unsqueeze(0)
                    left_indices_sent = left_indices[sent].unsqueeze(0)
                    aspect_double_idx_sent = aspect_double_idx[sent]
                    adjacent_sent = adjacent[sent][:text_len,:text_len]
                    text_indices_sent_rl = text_indices[sent][:text_len].unsqueeze(0)

                    text_vec_sent = criticModel.wordvector_find(text_indices_sent_rl, aspect_double_idx_sent.unsqueeze(0), text_len, aspect_len[sent].unsqueeze(0))

                    actions, states, Rinput, Rlength, aspect_last = rl.Sampling_RL(actorModel, criticModel, text_indices_sent_rl, text_vec_sent, aspect_double_idx_sent, Random=False)
                    adj = adjacent_sent[:].unsqueeze(0)
                    actions = torch.tensor(actions).cuda()

                    input_critic = [text_indices_sent_rl, aspect_indices_sent, left_indices_sent, adj, actions]

                    prediction, _ = criticModel(input_critic, scope = "target")

                    pred[sent] = prediction
                    
                n_test_correct += (torch.argmax(pred, -1) == t_targets).sum().item()
                n_test_total += len(pred)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    pred_all = pred
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    pred_all = torch.cat((pred_all, pred), dim=0)

        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(pred_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return test_acc, f1

    def run(self, repeats=3, alters=5):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        if not os.path.exists('A2RL/log/'):
            os.mkdir('A2RL/log/')

        f_out = open('A2RL/log/'+self.opt.model_name+'_'+self.opt.dataset+'_val.txt', 'w', encoding='utf-8')
        max_test_acc_avg = 0
        max_test_f1_avg = 0

        max_test_acc_pre_avg = 0
        max_test_f1_pre_avg = 0

        start = datetime.datetime.now()
        self.global_f1_all = 0.
        
        for i in range(repeats):
            print('repeat: ', (i+1))
            f_out.write('repeat: '+str(i+1) + '\n')
            self._reset_params()

            critic_path = 'A2RL/state_dict_temp/'+self.opt.dataset+'_'+str(self.opt.loss_punish_ratio)+'_critic.pkl'
            actor_path = 'A2RL/state_dict_temp/'+self.opt.dataset+'_'+str(self.opt.loss_punish_ratio)+'_actor.pkl'
            pretrain_path = 'A2RL/state_dict_temp/'+self.opt.dataset+'_'+str(self.opt.loss_punish_ratio)+'_pre.pkl'

            # 分类器热启动
            self.global_f1_pre = 0.
            self.global_f1 = 0.
            max_alter_acc = 0.
            max_alter_f1 = 0.
            
            print("critic pre training...")
            max_test_acc_pre, max_test_f1_pre = self._train(criterion, self.actorModel, self.criticModel, RL_train=False, LSTM_train=False)
            print('max_test_acc_pre: {0}     max_test_f1_pre: {1} '.format(max_test_acc_pre, max_test_f1_pre))
            f_out.write('max_test_acc_pre: {0}, max_test_f1_pre: {1}'.format(max_test_acc_pre, max_test_f1_pre) + '\n')

            print("critic pre loading...")
            self.criticModel.load_state_dict(torch.load(pretrain_path))
            
            for alter in range(alters):
                f_out.write('alter: '+str(alter+1) + '\n')
                print('alter: ', (alter + 1))
                # 交替训练 actor
                print("actor training...")
                self.global_f1 = 0.
                max_test_acc_rl, max_test_f1_rl = self._train(criterion, self.actorModel, self.criticModel, RL_train=True, LSTM_train=False)
                
                print("actor loading...")
                self.actorModel.load_state_dict(torch.load(actor_path))
                print('this alter: {0}_1     max_test_acc_rl: {1}     max_test_f1_rl: {2} '.format(alter + 1, max_test_acc_rl, max_test_f1_rl))

                # 交替训练 critic
                print("critic training...")
                self.global_f1 = 0.
                max_test_acc, max_test_f1 = self._train(criterion, self.actorModel, self.criticModel, RL_train=False, LSTM_train=True)

                print("critic loading...")
                self.criticModel.load_state_dict(torch.load(critic_path))
                
                print('this alter: {0}_2     max_test_acc: {1}     max_test_f1: {2}'.format(alter + 1, max_test_acc, max_test_f1))
                
                if max_test_f1_rl > max_alter_f1:
                    max_alter_acc = max_test_acc_rl
                    max_alter_f1 = max_test_f1_rl

                if max_test_f1 > max_alter_f1:
                    max_alter_acc = max_test_acc
                    max_alter_f1 = max_test_f1
                print('the whole alter: {0}    max_alter_acc: {1}     max_alter_f1: {2}  '.format(alter + 1, max_alter_acc, max_alter_f1))

                f_out.write('max_alter_acc: {0}, max_alter_f1: {1}'.format(max_alter_acc, max_alter_f1) + '\n')

            f_out.write('max_final_acc: {0}, max_final_f1: {1}'.format(max_alter_acc, max_alter_f1) + '\n')

            max_test_acc_pre_avg += max_test_acc_pre
            max_test_f1_pre_avg += max_test_f1_pre

            max_test_acc_avg += max_alter_acc
            max_test_f1_avg += max_alter_f1
            
            print('#' * 100)

        print("max_test_acc_pre_avg:", max_test_acc_pre_avg / repeats)
        print("max_test_f1_pre_avg:", max_test_f1_pre_avg / repeats)
        print("max_test_acc_avg:", max_test_acc_avg / repeats)
        print("max_test_f1_avg:", max_test_f1_avg / repeats)
        end = datetime.datetime.now()
        print(end - start)
        f_out.close()


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='asgcn', type=str)
    parser.add_argument('--dataset', default='rest15', type=str, help='twitter, rest14, lap14, rest15, rest16')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--save', default=True, type=bool)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--tau', default=0.1, type=float)
    parser.add_argument('--samplecnt', default=3, type=int)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--seed', default=776, type=int)
    parser.add_argument('--early_stop', default=2, type=int)
    parser.add_argument('--epsilon', default=0.1, type=float)
    parser.add_argument('--loss_punish_ratio', default=0.2, type=float)
    opt = parser.parse_args()

    model_classes = {
        'lstm': LSTM,
        'ascnn': ASCNN,
        'asgcn': ASGCN,
        'astcn': ASTCN,
    }
    input_colses = {
        'lstm': ['text_indices'],
        'ascnn': ['text_indices', 'aspect_indices', 'left_indices'],
        'asgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'astcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_tree'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.cuda.empty_cache()
    ins = Instructor(opt)
    ins.run()
