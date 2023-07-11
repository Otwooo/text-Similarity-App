import kss
from konlpy.tag import Okt
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

def cal_TF():
    pass

if __name__ == "__main__":
    s = "텍스트 전처리는 풀고자 하는 문제의 용도에 맞게 텍스트를 사전에 처리하는 작업입니다. 요리를 할 때 재료를 제대로 손질하지 않으면, 요리가 엉망이 되는 것처럼 텍스트에 제대로 전처리를 하지 않으면 뒤에서 배울 자연어 처리 기법들이 제대로 동작하지 않습니다. 이번 챕터에서는 텍스트를 위한 다양한 전처리 방법들에 대해서 다룹니다."
    pre1 = sentence_classification(s)
    print(pre1)  

    pre2 = noun_extraction(pre1)
    print(pre2)

