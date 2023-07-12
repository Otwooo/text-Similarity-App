from django.shortcuts import render, HttpResponse
from . import helpers

def index(request):
    return render(request, 'main/index.html')

def result(request):
    s = "텍스트 전처리는 풀고자 하는 문제의 용도에 맞게 텍스트를 사전에 처리하는 작업입니다. 요리를 할 때 재료를 제대로 손질하지 않으면, 요리가 엉망이 되는 것처럼 텍스트에 제대로 전처리를 하지 않으면 뒤에서 배울 자연어 처리 기법들이 제대로 동작하지 않습니다. 이번 챕터에서는 텍스트를 위한 다양한 전처리 방법들에 대해서 다룹니다."
    pre1 = helpers.sentence_classification(s)
    pre2 = helpers.noun_extraction(pre1)   
    TF = helpers.cal_TF(pre2) 

    return render(request, 'main/result.html', {"TF":TF})