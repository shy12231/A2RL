import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from copy import deepcopy
import numpy as np
import random
from layers.dynamic_rnn import DynamicLSTM
from .Layer2 import SimpleCat

def Sampling_RL(actor, critic, x, vector, aspect_idx, Random=True, epsilon=0.1):
    vector = vector.squeeze(0)
    start, end = aspect_idx[0].cpu().numpy(), (aspect_idx[1]).cpu().numpy()+1
    aspect_embed = vector[start: end]
    # aspect_embed = torch.max(aspect_embed, dim=0)[0].unsqueeze(0)
    aspect_embed = torch.mean(aspect_embed, dim=0).unsqueeze(0)
    current_lower_state = torch.zeros(1, 2 * 300).cuda()
    actions = []
    states = []
    for pos in range(len(vector)):
        predicted = actor.get_target_output(current_lower_state, vector[pos], aspect_embed, scope = "target")
        states.append([current_lower_state, vector[pos], aspect_embed])
        if pos >= start and pos < end:
            action = 1
        else:
            if Random:
                if random.random() > epsilon:
                    action = (0 if random.random() < float(predicted[0].item()) else 1)
                else:
                    action = (1 if random.random() < float(predicted[0].item()) else 0)
                # m = torch.distributions.Categorical(predicted.view(1,-1))
                # action = m.sample()
            else:
                action = torch.argmax(predicted).item()

        actions.append(action)
        if pos == start:
            aspect_last = torch.sum(torch.tensor(actions, dtype=torch.long) != 0, dim=-1).numpy()

        if action == 1:
            out_d, current_lower_state = critic.forward_lstm(current_lower_state, vector[pos], scope = "target")
    
    Rinput = []
    for idx, action in enumerate(actions):
        if action == 1:
            Rinput.append(int(x[0][idx].item()))
    Rlength = len(Rinput)
    Rinput = torch.tensor(Rinput).view(1,-1).cuda()
    
    return actions, states, Rinput, Rlength, aspect_last

class actor(nn.Module):
    def __init__(self, opt):
        super(actor, self).__init__()
        self.target_policy = policyNet(opt)
        self.active_policy = policyNet(opt)
        
    def get_target_logOutput(self, h, x):
        out = self.target_policy(h, x)
        logOut = torch.log(out)
        return logOut

    def get_target_output(self, h, x, aspect, scope):
        if scope == "target":
            out = self.target_policy(h, x, aspect)
        if scope == "active":
            out = self.active_policy(h, x, aspect)
        return out

    def get_gradient(self, h, x, aspect, reward, scope):
        if scope == "target":
            out = self.target_policy(h, x, aspect) # torch.Size([2, 1])
            logout = torch.log(out).view(-1)
            index = reward.index(0)
            index = (index + 1) % 2
            #print(out, reward, index, logout[index].view(-1), logout)
            #print(logout[index].view(-1))
            grad = torch.autograd.grad(logout[index].view(-1), self.target_policy.parameters()) # torch.cuda.FloatTensor(reward[index])
            #print(grad[0].size(), grad[1].size(), grad[2].size())
            #print(grad[0], grad[1], grad[2])
            grad[0].data = grad[0].data * reward[index]
            grad[1].data = grad[1].data * reward[index]
            grad[2].data = grad[2].data * reward[index]
            grad[3].data = grad[3].data * reward[index]
            #print(grad[0], grad[1], grad[2])
            return grad
        if scope == "active":
            out = self.active_policy(h, x)
        return out
    def assign_active_network_gradients(self, grad1, grad2, grad3, grad4):
        params = [grad1, grad2, grad3, grad4]
        i=0
        for name, x in self.active_policy.named_parameters():
            x.grad = deepcopy(params[i])
            i+=1

    def update_target_network(self, tau):
        params = []
        for name, x in self.active_policy.named_parameters():
            params.append(x)
        i=0
        for name, x in self.target_policy.named_parameters():
            x.data = deepcopy(params[i].data * (tau) + x.data * (1-tau))
            i+=1

    def assign_active_network(self):
        params = []
        for name, x in self.target_policy.named_parameters():
            params.append(x)
        i=0
        for name, x in self.active_policy.named_parameters():
            x.data = deepcopy(params[i].data)
            i+=1

class policyNet(nn.Module):
    def __init__(self, opt):
        super(policyNet, self).__init__()
        self.W1 = nn.Parameter(torch.cuda.FloatTensor(2 * opt.hidden_dim, 1).uniform_(-0.5, 0.5))
        self.W2 = nn.Parameter(torch.cuda.FloatTensor(opt.embed_dim, 1).uniform_(-0.5, 0.5))
        self.W3 = nn.Parameter(torch.cuda.FloatTensor(opt.embed_dim, 1).uniform_(-0.5, 0.5))
        self.b = nn.Parameter(torch.cuda.FloatTensor(1, 1).uniform_(-0.5, 0.5))

    def forward(self, h, x, aspect):
        h_ = torch.matmul(h.view(1,-1), self.W1)
        x_ = torch.matmul(x.view(1,-1), self.W2)
        aspect_ = torch.matmul(aspect.view(1,-1), self.W3)
        scaled_out = torch.sigmoid(h_ +  x_ + aspect_ + self.b)
        scaled_out = torch.clamp(scaled_out, min=1e-5, max=1 - 1e-5)
        scaled_out = torch.cat([1.0 - scaled_out, scaled_out],0)
        return scaled_out

class critic(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(critic, self).__init__()
        self.opt = opt
        self.target_pred = GCNClassifier(embedding_matrix, opt)
        self.active_pred = GCNClassifier(embedding_matrix, opt)

    def forward(self, inputs, scope):
        if scope == "target":
            out = self.target_pred(inputs)
        if scope == "active":
            out = self.active_pred(inputs)
        return out

    def assign_target_network(self):
        params = []
        for name, x in self.active_pred.named_parameters():
            params.append(x)
        i=0
        for name, x in self.target_pred.named_parameters():
            x.data = deepcopy(params[i].data)
            i+=1

    def update_target_network(self, tau):
        params = []
        for name, x in self.active_pred.named_parameters():
            params.append(x)
        i=0
        for name, x in self.target_pred.named_parameters():
            x.data = deepcopy(params[i].data * (tau) + x.data * (1-tau))
            i+=1

    def assign_active_network(self):
        params = []
        for name, x in self.target_pred.named_parameters():
            params.append(x)
        i=0
        for name, x in self.active_pred.named_parameters():
            x.data = deepcopy(params[i].data)
            i+=1

    def assign_active_network_gradients(self):
        params = []
        for name, x in self.target_pred.named_parameters():
            params.append(x)
        i=0
        for name, x in self.active_pred.named_parameters():
            x.grad = deepcopy(params[i].grad)
            i+=1
        for name, x in self.target_pred.named_parameters():
            x.grad = None

    def forward_lstm(self, hc, x, scope):
        if scope == "target":
            out, state = self.target_pred.getNextHiddenState(hc, x)
        if scope == "active":
            out, state = self.active_pred.getNextHiddenState(hc, x)
        return out, state

    def wordvector_find(self, x, aspect_double_idx, text_len, aspect_len):
        return self.target_pred.wordvector_find(x, aspect_double_idx, text_len, aspect_len)

class GCNClassifier(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(GCNClassifier, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        self.fc2 = nn.Linear(2*opt.hidden_dim, opt.hidden_dim)
        self.text_embed_dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(opt.embed_dim, opt.hidden_dim)

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj, actions = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len + aspect_len-1).unsqueeze(1)], dim=1)
        # print(actions)
        if actions is not None:
            text_indices = text_indices * actions
        
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len.cpu())

        if actions is not None:
            actions = actions.unsqueeze(1)
            dim = actions.size(0)
            a = actions.expand(dim, 600).to(torch.float32)
            text_out = text_out * a

        x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))

        x = self.mask(x, aspect_double_idx) # x通过两次图卷积,融合了句子的结构信息后,对除方面词外的词进行遮掩
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output, alpha

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):
                weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i,1]+1, text_len[i]):
                weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight, dtype=torch.float).unsqueeze(2).to(self.opt.device)
        return weight*x

    def position_embed(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):
                weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                weight[i].append(1)
            for j in range(aspect_double_idx[i,1]+1, text_len[i]):
                weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight, dtype=torch.float).unsqueeze(2).to(self.opt.device)
        return weight*x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i,1]+1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(2).to(self.opt.device)
        return mask*x

    def wordvector_find(self, x, aspect_double_idx, text_len, aspect_len):
        text = self.embed(x)
        text = self.text_embed_dropout(text)
        # text, (_, _) = self.text_lstm(text, text_len.cpu())
        # text = self.fc2(text)
        # text = self.position_embed(text, aspect_double_idx, text_len, aspect_len)
        return text

    def getNextHiddenState(self, hc, x):
        hidden = hc[0,0:self.opt.hidden_dim].view(1,1,self.opt.hidden_dim)
        cell = hc[0,self.opt.hidden_dim:].view(1,1,self.opt.hidden_dim)
        input = x.view(1,1,-1)
        out, hidden = self.lstm(input, [hidden, cell])
        hidden = torch.cat([hidden[0], hidden[1]], -1).view(1, -1)
        return out, hidden

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output
