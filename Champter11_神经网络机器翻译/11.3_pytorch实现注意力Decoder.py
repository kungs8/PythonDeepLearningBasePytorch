# -*- encoding: utf-8 -*-
"""
@File       : 11.3_pytorch实现注意力Decoder.py
@Time       : 2023/8/28 17:09
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
# 1. 构建Encoder
#     用pytorch 构建 Encoder比较简单，把输入语句的每个单词用`torch.nn.Embedding(m,n)`转换为词向量，然后通过一个编码器转换，
#     这里使用GRU循环网络，对每个输入字、编码器输出向量和隐藏状态，并将隐藏状态用于下一个输入字
import torch
from torch import nn

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)




# 2. 构建简单Decoder
# 构建一个简单的解释器，这个解释器这里只使用编码器的最后输出。这最后一个输出有时称为上下文向量，因为它从整个序列中编码上下文。该上下文向量用作解码器的初始隐藏状态。
# 在解码的每一步，解码器都被赋予一个输入指令和隐藏状态。初始输入指令字符串开始的<SOS>指令，第一个隐藏状态是上下文向量(编码器的最后隐藏状态)。
from torch.nn import functional as F
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(hidden_size, output_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size,output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



# 3. 构建注意力Decoder
# 这里使用的是 Bahdanau 注意力架构，主要有4层。嵌入层(Embedding Layer)将输入字转换为矢量，计算每个编码器输出的注意能量的层、RNN层和输出层。
class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(BahdanauAttnDecoderRNN, self).__init__()
        # 定义参数
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        # 定义层
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = GeneralAttn(hidden_size)
        self.gru = nn.GRU(hidden_size*2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        # 前向传播每次运行一个时间步，但使用所有的编码器输出
        # 获取当前词嵌入(last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1)  # S = 1*B*N
        word_embedded = self.dropout(word_embedded)

        # 计算注意力权重并使用编码器输出
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B*1*N

        # 把词嵌入与注意力context结合在一起，然后传入循环网络
        rnn_input = torch.cat((word_embedded, context), dim=2)
        output, hidden = self.gru(rnn_input, last_hidden)

        # 定义最后输出层
        output = output.squeeze(0)  # B*N
        output = F.log_softmax(self.out(torch.cat((output, context), dim=1)))

        # 返回最后输出，隐含状态及注意力权重
        return output, hidden, attn_weights