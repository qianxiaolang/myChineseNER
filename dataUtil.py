import math

import jieba
import numpy as np


def get_seg_features(sentence):
    seg_features=[]
    for word in jieba.cut(sentence):
        if len(word)==1:
            seg_features.append(0)
        else:
            temp=[2]*len(word)
            temp[0]=1
            temp[-1]=3
            seg_features.extend(temp)

    return seg_features

class DataManager(object):
    def __init__(self,train_data,batch_size):
        self.batch_data=self.sort_and_pad(train_data,batch_size)
        self.len_data=len(self.batch_data)

    def sort_and_pad(self,train_data,batch_size):
        ret=[]
        train_data=sorted(train_data,key=lambda x:len(x[0]))
        num_batch=int(math.ceil(len(train_data)/batch_size))
        for i in range(num_batch):
            batch_data=train_data[int(i*batch_size):int((i+1)*batch_size)]
            len_max=max([len(d[0]) for d in batch_data])
            temp_list=[]
            for j in range(len(batch_data)):
                data=batch_data[j]
                data[0].extend((len_max-len(data[3]))*['<PAD>'])
                data[1].extend((len_max-len(data[3]))*[0])
                data[2].extend((len_max - len(data[3])) * [0])
                data[3].extend((len_max - len(data[3])) * [0])
                temp=[]
                temp.append(data[1])
                temp.append(data[2])
                temp.append(data[3])
                temp_list.append(temp)

            ret.append(temp_list)

        return ret

    def iter_batch(self,shuffle=False):
        if shuffle:
            np.random.shuffle(self.batch_data)
        for i in range(self.len_data):
            yield self.batch_data[i]

    def iter_batch_index(self,index,shuffle=False):
        if shuffle:
            np.random.shuffle(self.batch_data)

        return np.array(self.batch_data[np.random.randint(min(3000,self.len_data))])

if __name__=='__main__':
    get_seg_features("我有一座冒险屋")