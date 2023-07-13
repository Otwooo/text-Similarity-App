from django.shortcuts import render, HttpResponse
from . import helpers
import copy

def index(request):
    return render(request, 'main/index.html')

def result(request):
    s = request.POST.get('text')    
    pre1 = helpers.sentence_classification(s)    
    noun = helpers.noun_extraction(pre1)    
    TF, TF_head = helpers.cal_TF(noun)             
    temp_TF = copy.deepcopy(TF)
    TF_IDF = helpers.cal_TF_IDF(noun, temp_TF, TF_head)
    cos = helpers.cosine_sim(TF) 
    jac = helpers.jaccard(TF)
    Euc = helpers.Euclidean(TF)
    man = helpers.manhattan(TF)

    print_TF = []
    for i in TF:
        print_TF.append(i[0:31])
    print_TF_IDF = []
    for i in TF:
        print_TF_IDF.append(i[0:31])
    return render(request, 'main/result.html', {"TF":print_TF, "TF_head":TF_head[0:31], "TF_IDF":print_TF_IDF, "cos":cos, "jac":jac, "Euc":Euc, "man":man})    