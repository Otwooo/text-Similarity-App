import kss # 문장 분류
from konlpy.tag import Okt # 명사 추출
from collections import Counter # 명사 빈도수 계산
import math
import copy
import numpy as np
from numpy import dot
from numpy.linalg import norm

tagger = Okt()

def sentence_classification(sentence):
    result = []
    for sent in kss.split_sentences(sentence):
        result.append(sent)
    return result

def noun_extraction(sentence):
    global tagger

    result = []
    for i in range(len(sentence)):
        result.append(tagger.nouns(sentence[i]))
    return result

def cal_TF(noun):

    word = []
    for i in range(len(noun)):
        for j in range(len(noun[i])):
            word.append(noun[i][j])

    count = Counter(word)
    
    noun_list = count.most_common(3000)
    # for v in noun_list:
    #     print(v)

    TF_head = []
    for i in range(len(noun_list)):
        TF_head.append(noun_list[i][0])

    TF = [[0 for i in range(len(noun_list))] for i in range(len(noun))]
    for i in range(len(noun)):
        for j in range(len(noun_list)):
            TF[i][j] = noun[i].count(noun_list[j][0])
    
    return TF, TF_head

def cal_DF(noun, word):
    result = 0
    for i in range(len(noun)):
        if word in noun[i]:
            result += 1

    return result

def cal_TF_IDF(noun, li, TF_head):
    for i in range(len(li)):
        for j in range(len(li[i])):
            li[i][j] = round(li[i][j]*math.log(len(noun)/(cal_DF(noun, TF_head[j]))), 3)
    
    return li

def cosine_sim(TF):
    result = [[] for i in range(len(TF))]  
    for i in range(len(TF)):
        for j in range(len(TF)):            
            A = np.array(TF[i])
            B = np.array(TF[j])
            result[i].append(round(dot(A, B)/(norm(A)*norm(B)), 3))

    return result

def jaccard(TF):
    result = [[] for i in range(len(TF))]
    for i in range(len(TF)):
        for j in range(len(TF)):
            s = 0
            for x in range(len(TF[0])):
                if TF[i][x] == TF[j][x]: 
                    s+= 1
            result[i].append(round(s/len(TF[0]), 3))

    return result

def Euclidean(TF):
    result = [[] for i in range(len(TF))]
    for i in range(len(TF)):
        for j in range(len(TF)):            
            A = np.array(TF[i])
            B = np.array(TF[j])
            result[i].append(round(np.sqrt(np.sum((A-B)**2)), 3))
    
    return result

def manhattan(TF):
    result = [[] for i in range(len(TF))]
    for i in range(len(TF)):
        for j in range(len(TF)):            
            A = np.array(TF[i])
            B = np.array(TF[j])
            distance = 0
            for x in range(len(A)):
                 distance += abs(A[x] - B[x])
            result[i].append(distance)

    return result

if __name__ == "__main__":
    s = """텍스트 전처리는 풀고자 하는 문제의 용도에 맞게 텍스트를 사전에 처리하는 작업입니다. 요리를 할 때 재료를 제대로 손질하지 않으면, 요리가 엉망이 되는 것처럼 텍스트에 제대로 전처리를 하지 않으면 뒤에서 배울 자연어 처리 기법들이 제대로 동작하지 않습니다. 이번 챕터에서는 텍스트를 위한 다양한 전처리 방법들에 대해서 다룹니다."""
    pre1 = sentence_classification(s)    
    noun = noun_extraction(pre1)    
    TF, TF_head = cal_TF(noun)                 
    temp_TF = copy.deepcopy(TF)
    TF_IDF = cal_TF_IDF(noun, temp_TF, TF_head)    
    cos = cosine_sim(TF)
    jac = jaccard(TF)
    Euc = Euclidean(TF)
    man = manhattan(TF)

    print("noun :", noun)
    print("----"*20)
    print("TF :", TF)
    print("----"*20)
    print("TF_head :", TF_head)
    print("----"*20)
    print("TH_IDF :", TF_IDF)
    print("----"*20)
    print("cos :", cos)
    print("----"*20)
    print("jaccard :", jac)
    print("----"*20)
    print("Euclidean :", Euc)
    print("----"*20)
    print("manhattan :", man)