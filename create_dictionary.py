from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json


# 读取数据，将数据转成词语列表.
def read_data(filename):
    with open(filename, encoding="utf-8") as f:
        data = f.read()
    data = list(data)
    return data


# 生成字典dictionary和reversed_dictionary
def build_dataset(words, n_words):
    
    count = [['UNK', -1]]                                 # 创建列表count,第一个元素是['UNK', -1],之后根据单词频数高低添加(extend)到count
    count.extend(collections.Counter(words).most_common(n_words - 1))  # 使用collections.Counter统计单词列表中单词的频数
                                                                       # most_common方法取频数高的单词 
    dictionary = dict()                                                # 创建一个字典，将高频单词放入dictionary中，以便快速查询
    for word, _ in count:
        dictionary[word] = len(dictionary)                             # [word](key)是对应的单词，值(value)是出现次数的排名

    data = list()
    unk_count = 0
    for word in words:                                                # 遍历单词列表
        index = dictionary.get(word, 0)                                # 如果word是dictionary的key，获取对应的value，如果没有value就是0
        if index == 0:   
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count

    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))  # 将key和value互换生成新的reversed_dictionary
    
    return data, count, dictionary, reversed_dictionary    # 返回转换后的编码(data),每个单词的频数统计(count),字典(dictionary)及反转形式


filename = './QuanSongCi.txt'

vocabulary = read_data(filename)
print('Data size', len(vocabulary))

vocabulary_size = 5000

data, count, dictionary, reversed_dictionary = build_dataset(vocabulary,vocabulary_size - 1)

#json.dump()函数的使用，保存json文件
with open("./dictionary.json","w",encoding='utf-8') as f:
    json.dump(dictionary,f)

with open("./reverse_dictionary.json","w",encoding='utf-8') as f:
    json.dump(reversed_dictionary,f)

