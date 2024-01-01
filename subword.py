import math
import time
import codecs
import os
import re
import collections
import pickle

def get_subwords(data):
    """
    统计子词以及对应的词频
    """
    subwords = collections.defaultdict(int)
    for word, freq in data.items():
        for subword in word.split():
            subwords[subword] += freq

    return subwords

def get_pair_with_frequency(data):
    """
    获取子词对以及子词集合
    """
    pairs = collections.defaultdict(int)
    for word, freq in data.items():
        sub_words = word.split()
        for i in range(len(sub_words)-1):
            pair = (sub_words[i],sub_words[i+1])
            pairs[pair] += freq
    return pairs

def merge_data_with_pair(pair, data):
    """
    将语料中的最高频子词对进行合并
    输入：
        - pair: 最高频子词词对
        - data: 字典形式，统计好的输入语料
    """
    result = {}
    bigram = re.escape(' '.join(pair)) # 替换 X Y 
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)') # 匹配任意的X Y
    for word in data:
        merged_word = p.sub(''.join(pair), word) # 替换
        result[merged_word] = data[word]
        # if merged_word != word:
        #     print('ss')
    return result

def build_vocab(train_data, num_merges):
    """
    根据训练语料构建词表
    输入：
        - train_data: 字典形式，统计好的输入语料
        - num_merges: 迭代次数
    """

    # 初始化词表
    f1 = open('merge_word_log.txt','w',encoding='utf-8')
    subwords = get_subwords(train_data) #统计词表
    bpe_vocab = set(subwords.keys()) # 初始化bpe词表 
    print(bpe_vocab, len(bpe_vocab))
    i = 1
    # 逐步生成词表
    for _ in range(num_merges):
        # 根据语料统计相邻子词对的词频
        pairs = get_pair_with_frequency(train_data)
        # 取频率最大的子词对, 如果pairs 为空或子词对的最大频次为1，则停止
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        if pairs[best_pair] == 1:
            break
        # 遍历语料单词，合并单词内的一对文本
        train_data = merge_data_with_pair(best_pair, train_data)
        # 将子词加入词表中
        merged_word = "".join(best_pair)
        bpe_vocab.add(merged_word)
        # 删除子词：如果合并的两部分在语料中不再存在，则删去
        subwords = get_subwords(train_data)
        if best_pair[0] not in subwords:
            bpe_vocab.remove(best_pair[0])
        if best_pair[1] not in subwords and best_pair[0] != best_pair[1]:
            bpe_vocab.remove(best_pair[1])

        print("Iter - {}, 最高频子词对: {}".format(i, best_pair))
        f1.write(''.join(best_pair))
        # print("训练数据: ", train_data)
        print("词表长度: {}".format(len(bpe_vocab)))
        i += 1
    f1.close()    
    return bpe_vocab

if __name__ == '__main__':
    
    with open("./data/199801.txt", "r", encoding="gbk") as f:
        content = [_.strip() for _ in f.readlines() if _.strip()]
    word_set = set()
    train_data = {} # 统计分词，以分词后的词单位进行bpe算法
    for line in content[:int(len(content)*0.9)]:
        for item in line.split()[1:]:
            word = ' '.join(item.split("/")[0]) # 首先以单个字符为基本单位
            if word not in word_set:
                word_set.add(word)
                train_data[word] = 1
            else:
                train_data[word] += 1    

    bpe_vocab = build_vocab(train_data, 1000)
    bpe_vocab_file = open('bpe_vocab.pickle','wb')
    pickle.dump(bpe_vocab,bpe_vocab_file)
    bpe_vocab_file.close()