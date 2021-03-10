##停用词列表包括符号
import jieba
import math
import numpy as np

def stopword_list(file='stopword.txt'):
    stopwords = [line.strip() for line in open(file,encoding='UTF-8').readlines()]
    return stopwords

def word_cut(sentence,stopwords):
    ret=[]
    result = jieba.cut(sentence.strip())
    for word in result:
        if word not in stopwords:
            ret.append(word)
    return ret


def read_file(file,word_func=word_cut):
    ret=[]
    with open(file,encoding="UTF-8") as f:
        for line in f.readlines():
            ret.extend(word_func(line,stopword_list()))
    return ret

def get_all_vocabulary(list_files):###获得语料库中的词汇
    vocabulary=[]
    for i in list_files:
        vocabulary.append(read_file(i))
    return vocabulary

def get_setof_vocabulary(vocabulary,setize=1):
    ret=[]
    for i in vocabulary:
    
        ret.extend(i)
    return list(set(ret)) if setize else ret

def tf_calcus(index,vocabulary,word_set):##计算tf
    tf_list=[]
    word=vocabulary[index]
    
    vector=vector_trans(word,word_set)
    
    sum_num=sum(vector)
    for i in vector:
        tf_list.append(i/sum_num)######TF=某个词的次数/该文章中最多的次数
    return tf_list

def vector_trans(wordlist,word_set,mode=0):###将单词转化为向量，基于所有单词的集合
    ret=[]
    if mode==0:
        for i in word_set:
            count=wordlist.count(i)
            ret.append(count if count else 0)
        return ret



def idf_calcus(index,vocabulary,word_set):###IDF=log(语料库文档总数=len(vocabulary)/包含该词的文档数+1)=unsetof_vocabulary
    idf_list=[]
    word=vocabulary[index]
    allword=word_set

    for i in allword:
        
        count=0
        for wordgroup in vocabulary:##tf*idf，tf=0,相乘=0,无所谓idf值
            
            count+=1 if i in wordgroup else 0
        
        idf_list.append(math.log(len(vocabulary)/(count+1)))
    return idf_list

def tf_idf_calcus(tf,idf):
    ret=[]

    if not tf and not idf:
        
        return ret
    num=len(tf)
    num1=len(idf)
    
    if len(tf)==len(idf) :
        
        for i in range(num):
            ret.append(tf[i]*idf[i])

    return ret

def compare_two(doc_fir,doc_sec,tf_idf_all):
    
    return cosine_similarity(tf_idf_all[doc_fir],tf_idf_all[doc_sec])

def cosine_similarity(x, y, norm=False):
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return 0.5 * cos + 0.5 if norm else cos

if __name__=="__main__":
    
    list_files=["101.txt","102.txt"]
    doc_nums=len(list_files)
    sep_word_list=get_all_vocabulary(list_files)
    #print(sep_word_list)
    setof_vocabulary=get_setof_vocabulary(sep_word_list)
    #print(setof_vocabulary)
    #unsetof_vocabulary=get_setof_vocabulary(sep_word
    # _list,0)

    tf_idf_all=[]
    for i in range(doc_nums):
        print(tf_calcus(i,sep_word_list,setof_vocabulary),idf_calcus(i,sep_word_list,setof_vocabulary))
        
        temp=tf_idf_calcus(tf_calcus(i,sep_word_list,setof_vocabulary),idf_calcus(i,sep_word_list,setof_vocabulary))
        tf_idf_all.append(temp)
    
    
    for i in range(doc_nums-1):
        for j in range(i+1,doc_nums):
            print("文档{}和{}之间的相似度为{:%}".format(list_files[i],list_files[j],compare_two(i,j,tf_idf_all)))
    
    
            


        
