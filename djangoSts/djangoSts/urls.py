"""djangoSts URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
import os
import base64
from django.contrib import admin
from django.urls import path
from django.http import HttpResponse
from django.http import HttpRequest
import nltk
import json
import argparse
from djangoSts.Mconfig import * 
from djangoSts.origin.train import main as originTrain
from djangoSts.mean.train import main as meanTrain
from djangoSts.zero.train import main as zeroTrain
from djangoSts.mean_zero.train import main as meanZeroTrain
from struct import pack, unpack

def process_sentence(sent, max_seq_len):
    '''process a sentence using NLTK toolkit'''
    return nltk.word_tokenize(sent)[:max_seq_len]
# print(11111)
def getTxtSim(request):
    output = {'error':'no'}
    if len(request.body) == 0:
        output['error'] = 'no input'
        return HttpResponse(json.dumps(output))
    input = json.loads(request.body)
    if 'text1' not in input or 'text2' not in input:
        output['error'] = 'text1 or text2 not in input'
        return HttpResponse(json.dumps(output))
    input['text1'] = process_sentence(input['text1'], 1000000000000000000000)
    input['text2'] = process_sentence(input['text2'], 1000000000000000000000)
    output = {'error':'no','similary':funTxt(input)}
    return HttpResponse(json.dumps(output))

def funTxt(input):
    allCommands = []
    f = open('file.txt','w')
    avg = 0
    for i,command in enumerate(commands):
        for j,task in enumerate(tasks):
            commands[i]["eval_model"] = tasks[j] + models[i] + model_name
            if commands[i]['eval_model'].find('origin') != -1:
                res = originTrain(commands[i],([input['text1'],input['text1']],[input['text2'],input['text2']],[1,1]))[0]/5*100
                avg += res
                f.write('origin-'+models[i][:-1]+','+str(res)+'\n')
            elif commands[i]['eval_model'].find('mean') != -1 and commands[i]['eval_model'].find('mean_zero') == -1:
                res = meanTrain(commands[i],([input['text1'],input['text1']],[input['text2'],input['text2']],[1,1]))[0]/5*100
                avg += res
                f.write('mean-'+models[i][:-1]+','+str(res)+'\n')
            elif commands[i]['eval_model'].find('zero') != -1 and commands[i]['eval_model'].find('mean_zero') == -1:
                res = zeroTrain(commands[i],([input['text1'],input['text1']],[input['text2'],input['text2']],[1,1]))[0]/5*100
                avg += res
                f.write('zero-'+models[i][:-1]+','+str(res)+'\n')
            elif commands[i]['eval_model'].find('mean_zero') != -1:
                res = meanZeroTrain(commands[i],([input['text1'],input['text1']],[input['text2'],input['text2']],[1,1]))[0]/5*100
                avg += res
                f.write('mean_zero-'+models[i][:-1]+','+str(res)+'\n')
    # print(args)
    avg /= len(tasks)*len(models)
    f.write('score:'+str(avg)+'\n')
    f.close()
    f = open('file.txt','rb')
    base64_str = base64.b64encode(f.read())
    f.close()
    return str(base64_str)[2:-1]

def funFile(input):
    allCommands = []
    f = open('file2.txt','w')
    avg = [0 for _ in input['text1']]
    presim = [1 for _ in input['text1']]
    for i,command in enumerate(commands):
        for j,task in enumerate(tasks):
            commands[i]["eval_model"] = tasks[j] + models[i] + model_name
            if commands[i]['eval_model'].find('origin') != -1:
                res = originTrain(commands[i],(input['text1'],input['text2'],presim))
                for x in range(len(res)):
                    res[x] = res[x]/5*100
                    avg[x] += res[x]
                f.write('origin-'+models[i][:-1]+','+str(res)+'\n')
            elif commands[i]['eval_model'].find('mean') != -1 and commands[i]['eval_model'].find('mean_zero') == -1:
                res = meanTrain(commands[i],(input['text1'],input['text2'],presim))
                for x in range(len(res)):
                    res[x] = res[x]/5*100
                    avg[x] += res[x]
                f.write('mean-'+models[i][:-1]+','+str(res)+'\n')
            elif commands[i]['eval_model'].find('zero') != -1 and commands[i]['eval_model'].find('mean_zero') == -1:
                res = zeroTrain(commands[i],(input['text1'],input['text2'],presim))
                for x in range(len(res)):
                    res[x] = res[x]/5*100
                    avg[x] += res[x]
                f.write('zero-'+models[i][:-1]+','+str(res)+'\n')
            elif commands[i]['eval_model'].find('mean_zero') != -1:
                res = meanZeroTrain(commands[i],(input['text1'],input['text2'],presim))
                for x in range(len(res)):
                    res[x] = res[x]/5*100
                    avg[x] += res[x]
                f.write('mean_zero-'+models[i][:-1]+','+str(res)+'\n')
    for i in range(len(avg)):
        avg[i] /= len(tasks)*len(models)
    f.write('score,'+str(avg)+'\n')
    f.close()
    f = open('file2.txt','rb')
    base64_str = base64.b64encode(f.read())
    f.close()
    return str(base64_str)[2:-1]

def getFileSim(request):
    output = {'error':'no'}
    if len(request.body) == 0:
        output['error'] = 'no input'
        return HttpResponse(json.dumps(output))
    input = json.loads(request.body)
    
    input['text1'] = base64.b64decode(input['text1'])[15:].decode().split('\n')
    input['text2'] = base64.b64decode(input['text2'])[15:].decode().split('\n')

    for i,x in enumerate(input['text1']):
        input['text1'][i] = process_sentence(input['text1'][i], 1000000000000000000000)
    for i,x in enumerate(input['text2']):
        input['text2'][i] = process_sentence(input['text2'][i], 1000000000000000000000)
    input['text1'] = [x for x in input['text1'] if len(x) > 0]
    input['text2'] = [x for x in input['text2'] if len(x) > 0]
    if 'text1' not in input or 'text2' not in input:
        output['error'] = 'text1 or text2 not in input'
        return HttpResponse(json.dumps(output))
    output = {'error':'no','similary':funFile(input)}
    return HttpResponse(json.dumps(output))

urlpatterns = [
    path('getFileSim/',getFileSim),
    path('getTxtSim/', getTxtSim)
]
