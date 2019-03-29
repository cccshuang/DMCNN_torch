import torch
import torch.nn as nn
import numpy as np


class dmcnn_t(nn.Module):
    def __init__(self, config):
        super(dmcnn_t, self).__init__()
        self.config = config
        self.keep_prob = 0.5

        self.char_inputs = None             # [batch, char_dim] 句子
        self.trigger_inputs = None          # [batch] 真实的trigger种类
        self.pf_inputs = None
        self.lxl_inputs = None               # [batch, sen
        self.masks = None                   # [batch, sen_len-2]    用于pooling   trigger位置之前值为1，trigger之后，填充部分之前为2，填充部分为0
        self.cuts = None                    # [batch, 1]    trigger位置
        
        self.char_lookup = nn.Embedding(self.config.num_char, self.config.char_dim)      # [20136, 100] word2vec
        self.pf_lookup = nn.Embedding(self.config.batch_t, self.config.pf_t)               # [batch, pf_dim]
        # self.init_word_weights()
        # self.init_pf_weights()

        self.conv = nn.Conv1d(self.config.char_dim+self.config.pf_t, self.config.feature_t, self.config.window_t, bias=True)
        self.L = nn.Linear(2*self.config.feature_t + 3*self.config.char_dim, self.config.num_t, bias=True)
        self.dropout = nn.Dropout(p=self.keep_prob)
        self.loss = nn.CrossEntropyLoss()

    def init_word_weights(self):
        self.char_lookup.weight.data.copy_(torch.from_numpy(self.config.emb_weights))

    def init_pf_weights(self):
        nn.init.xavier_uniform_(self.pf_lookup.weight.data)

    def pooling(self, conv):
        mask = np.array([[0, 0], [0, 1], [1, 0]])
        mask_emb = nn.Embedding(3, 2).cuda()
        mask_emb.weight.data.copy_(torch.from_numpy(mask))
        mask = mask_emb(self.masks)                         # conv [batch, sen-2, feature]   mask [batch, sen-2, 2]
        pooled, _ = torch.max(torch.unsqueeze(mask*100, dim=2) + torch.unsqueeze(conv, dim=3), dim=1)
        pooled -= 100
        pooled = pooled.view(self.config.batch_t, -1)
        return pooled

    def forward(self):
        x = torch.cat((self.char_lookup(self.char_inputs), self.pf_lookup(self.pf_inputs)), dim=-1)
        y = self.char_lookup(self.lxl_inputs).view(self.config.batch_t, -1)
        x = torch.tanh(self.conv(x.permute(0, 2, 1)))       # [batch, feature, sen-2]
        x = x.permute(0, 2, 1)                              # [batch, sen-2, feature]
        x = self.pooling(x)                                 # [batch, 2*feature]
        x = torch.cat((x, y), dim=-1)                       # [batch, 2*feature+3*char]
        # x = self.dropout(x)
        x = self.L(x)                                       # [batch, trigger]
        loss = self.loss(x, self.trigger_inputs)
        _, maxes = torch.max(x, dim=1)
        return loss, maxes






