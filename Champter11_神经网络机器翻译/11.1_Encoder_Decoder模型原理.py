# -*- encoding: utf-8 -*-
"""
@File       : 11.1_Encoder_Decoder模型原理.py
@Time       : 2023/8/18 18:12
@Author     : yanpenggong
@Email      : yanpenggong@163.com
@Version    : 1.0
@Copyright  : 侵权必究
"""
神经网络机器翻译(Neural Machine Translation, NMT)是一种机器翻译方法。
相比较传统的统计机器翻译(Statistical Machine Translation, SMT)而言，NMT能够训练从一个序列映射到另一个序列的神经网络，输出的可以是一个变长的序列，
在翻译、对话和文字概括方面已获得非常好的效果。
NMT其实是一个Encoder-Decoder系统，Encoder把源语言序列进行编码，并提取源语言中信息，通过Decoder再把这种信息转换到另一种语言即目标语言中来，从而完成对语言的翻译。

1. Encoder-Decode(编-解码器)模型原理
目前，机器翻译、文本摘要、语音识别等一般采用带注意力(Attention)的模型，它是对Encoder-Decoder模型的改进版本。
Encoder-Decoder亦称Seq2Seq模型。
实例：
    从左到右，可以这么理解：从左到右，看作适合处理由一个句子(或文章)生成的另外一个句子(或文章)的通用处理模型。假设这句子对<X,Y>，
    目标：输入句子X，通过Encoder-Decoder框架来生成目标句子<X，Y>。X、Y可以是同一种语言，也可以是不同的语言，X、Y由各自的单词序列组成。
Encoder：对输入句子X进行编码，将输入句子通过非线性变换转化为中间语义C=f(x1,x2,...,xm)
Decoder: 根据句子X的中间语义C和之前已经生成的历史信息(y1,y2,...,y_{i-1})来生成i时刻要生成的单词yi=g(C, y1,y2,...,yi-1)，每个yi都这么依次产生。
即：可以看成整个系统根据输入句子X生成了目标语句Y。
Encoder-Decoder框架具体使用模型：CNN/RNN/BiRNN/GRU/LSTM/Deep LSTM 等，而且变化组合非常多。

Encoder-Decoder 模型应用非常广泛，其应用场景也非常多，比如：
    - 机器翻译：<X,Y>就是对应不同语言的句子，如X是英语句子，Y就是对应的中文句子翻译
    - 文本摘要：X就是一篇文章，Y就是对应的摘要
    - 对话机器人：X是某人的一句话，Y就是对话机器人 应答
框架的缺点：
    生成的句子中每个词采用的中间语言编码是相同的，都是C。在语句比较短的时候，性能还可以，但句子稍微长一些，生成的句子就不尽如人意
    y1 = g(C)
    y2=g(C,y1)
    y3=g(C,y1,y2)
缺点解决办法：
    使用Attention框架机制，在C上做一些处理。