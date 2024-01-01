import math
import time
import codecs
import os
import re
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import classification_report
class WordSplitor(object):
    """分词模块类
    新建一个类对象用于对中文语句串进行分词
    """
    def __init__(self, user_dict=None):
        self.user_dict = user_dict
        self.dic_list = []
        self.max_length = 0
        if self.user_dict is not None:
            self.dic_list = self.__load_words(self.user_dict)
            self.max_length = len(sorted(self.dic_list, key=lambda x: len(x))[-1])

    def __load_words(self, fpath):
        dic_list = []
        with codecs.open(fpath, encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                dic_list.append(line)
        print("======== Loading Dict Finished ========")
        return dic_list

    def cut(self, sentence):
        """最大正向匹配法分词
        Args:
            sentence (str): 分词语句串

        Returns:
            list: 分词后得到的列表 每个元素是分词后的各个部分
        """
        ans_forward = []
        len_row = len(sentence)
        while len_row > 0:
            divide = sentence[0:self.max_length]
            while divide not in self.dic_list:
                if len(divide) == 1:
                    break
                divide = divide[0:len(divide)-1]
            ans_forward.append(divide)
            sentence = sentence[len(divide):]
            len_row = len(sentence)
        return ans_forward

def training_CRF():
    # 读取分词语料库，并转为CRF++工具接受的输入格式
    train_data = []
    test_data = []
    # 人民日报语料
    with open("./data/199801.txt", "r", encoding="gbk") as f:
        content = [_.strip() for _ in f.readlines() if _.strip()]

    # 训练数据
    for line in content[:int(len(content)*0.9)]:
        sent_tag = []
        for item in line.split()[1:]:
            word = item.split("/")[0]
            postag = item.split("/")[1]
            if len(word) == 1:
                sent_tag.append((word[0], 'n' ,'S'))
            else:
                sent_tag.append((word[0], 'n' ,'B'))
                if len(word) > 2:
                    for m in range(1, len(word) - 1):
                        sent_tag.append((word[m], 'n' ,'M'))
                sent_tag.append((word[-1], 'n' ,'E'))
        train_data.append(sent_tag)

    # 测试数据
    for line in content[int(len(content)*0.9)+1:]:
        sent_tag = []
        for item in line.split()[1:]:
            word = item.split("/")[0]
            postag = item.split("/")[1]

            if len(word) == 1:
                sent_tag.append((word[0], 'n' ,'S'))
            else:
                sent_tag.append((word[0], 'n' ,'B'))
                if len(word) > 2:
                    for m in range(1, len(word) - 1):
                        sent_tag.append((word[m], 'n' ,'M'))
                sent_tag.append((word[-1], 'n' ,'E'))
        test_data.append(sent_tag)
    
    with open("crf_train.txt", "w", encoding="utf-8") as f:
        for line in train_data:
            for word_tuple in line:
                f.write("\t".join(list(word_tuple))+"\n")
            f.write("\n")

    with open("crf_predict.txt", "w", encoding="utf-8") as f:
        for line in test_data:
            for word_tuple in line:
                f.write("\t".join(list(word_tuple))+"\n")
            f.write("\n")

    os.system('crf_learn template crf_train.txt crf_model')
    
    
def read_corpus_FMM():
    
    # 读取训练集的词典
    with open("./data/199801.txt", "r", encoding="gbk") as f:
        content = [_.strip() for _ in f.readlines() if _.strip()]

    dict_set = set()
    dict_list = []
    for line in content[:int(len(content)*0.9)]:
        for item in line.split()[1:]:
            word = item.split("/")[0]
            if word not in dict_set:
                dict_set.add(word)
                dict_list.append(word)
                
    with open("mydict.txt", "w", encoding="utf-8") as f:          
        for word in dict_list:
            f.write(word+'\n')

def model_eval_FMM():
    
    # 转化为与CRF中的一样表示
    
    ws = WordSplitor('mydict.txt')
    with open("./data/199801.txt", "r", encoding="gbk") as f:
        content = [_.strip() for _ in f.readlines() if _.strip()]
    # 测试数据
    y_true = []
    y_pred = []
    index = 0
    start_time = time.time()
    for line in content[int(len(content)*0.9)+1:]:
        index += 1
        sentence = ''
        sentence_true = []
        print("{}/{}".format(index, int(len(content)*0.1)))
        for item in line.split()[1:]:
            word = item.split("/")[0]
            sentence_true.append(word)
            sentence = sentence + word
            if len(word) == 1:
                y_true.append(list('S'))
            else:
                y_true.append(list('B'))
                if len(word) > 2:
                    for m in range(1, len(word) - 1):
                        y_true.append(list('M'))
                y_true.append(list('E'))
        cut_word = ws.cut(sentence)
        for cell in cut_word:
            if len(cell) == 1:
                y_pred.append(list('S'))
            else:
                y_pred.append(list('B'))
                if len(cell) > 2:
                    for m in range(1, len(cell) - 1):
                        y_pred.append(list('M'))
                y_pred.append(list('E'))   
                        
    print("accuary: ", accuracy_score(y_true, y_pred))
    print("p: ", precision_score(y_true, y_pred))
    print("r: ", recall_score(y_true, y_pred))
    print("f1: ", f1_score(y_true, y_pred))
    print("classification report: ")
    print(classification_report(y_true, y_pred))
    
    end_time = time.time()
    print('running time:{}s'.format(end_time - start_time))
                
def model_eval():
    
    with open("crf_pred.txt", "r", encoding='utf-16') as f:
        content = [_.strip() for _ in f.readlines()]

    y_pred = []
    y_true = []
    for line in content:
        if line:
            y_pred.append(list(line.split("\t")[-1]))
            y_true.append(list(line.split("\t")[-2]))

    print("accuary: ", accuracy_score(y_true, y_pred))
    print("p: ", precision_score(y_true, y_pred))
    print("r: ", recall_score(y_true, y_pred))
    print("f1: ", f1_score(y_true, y_pred))
    print("classification report: ")
    print(classification_report(y_true, y_pred))       

def predict_CRF(text):

    # 生成待预测的文本
    with open("predict.data", "w", encoding="utf-8") as g:
        for char in text:
            g.write("%s\tn\tB-Char\n" % char)

    # 利用CRF模型，调用命令行进行预测
    os.system("crf_test -m crf_model predict.data > predict_new.txt")

    # 处理预测后的进行，并将其加工成中文分词后的结果
    with open("predict_new.txt", "r", encoding="utf-8") as f:
        content = [_.strip() for _ in f.readlines()]

    predict_tags = []
    for line in content:
        predict_tags.append(line.split("\t")[-1])

    words = []
    for i in range(len(predict_tags)):
        word = ""
        if predict_tags[i] == "B":
            word += text[i]
            j = i + 1
            while j < len(text) and predict_tags[j] != "E":
                word += text[j]
                j += 1
            word += text[j]
            
        if predict_tags[i] == 'S':
            word += text[i]
            i += 1

        if word:
            words.append(word)

    
    return "/".join(words)

def check_word(word,ws):
    return word in ws.dic_list

def test_sentense():
    text1 = "李荣浩10岁时得到了人生中的第一把吉他。"
    text2 = '在没有老师的情况下，他通过卡带和教材自学。'
    text3 = '一场旷世之战落幕，然而却是留下了一个满目疮痍的中州，原本的繁华不在，甚至整个中州，都是在此刻被一分为二。'
    text4 = '在魂界的中央位置，联军发现了一个近十万丈庞大的血池，其中的血液粘稠无比。'
    texts = [text1,text2,text3,text4]
    ws = WordSplitor('mydict.txt')
    for text in texts:
        crf_cutwords = predict_CRF(text)
        cut_words = ws.cut(text)
        print('原文:',text)
        print('CRF分词结果:',crf_cutwords)
        print('FMM分词结果:','/'.join(cut_words))    

if __name__ == '__main__':
    
    # flag = 1
    # if flag:
    #     training_CRF()
    #     os.system('crf_test -m crf_model crf_predict.txt > crf_pred.txt')
    #     model_eval()
    # else:
    #     read_corpus_FMM()
    #     model_eval_FMM()
    
    # test_sentense()
    ws = WordSplitor('mydict.txt')
    print('s')
            