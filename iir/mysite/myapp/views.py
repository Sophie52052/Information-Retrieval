from django.shortcuts import render
from django.http import JsonResponse
from django.http import HttpResponse
from django.core import serializers
import json
from django.http import HttpResponseRedirect
from .forms import UploadFileForm
from .forms import handle_uploaded_file
from . import fulltext_xml as ft_xml
from . import fulltext_json as ft_json
from django.shortcuts import render_to_response
import numpy as np
# Create your views here.

def index(request):
    
    return render(request, 'myapp/index.html', locals())

    

def compare(request):
    print(request.POST['filename'])

    print(request.POST['keyword'])
    per_detail=ft_xml.process(request.POST['filename'],request.POST['keyword'])
    per_detail_Z=ft_xml.process(request.POST['filename'],"zika")
    
    return render(request,'myapp/compare.html', locals())


def mark_span(o_string,keyword):
    keyword=keyword.lower()
    rep='<span style="color:orange;">'+keyword+'</span>'
    o_string=o_string.replace(keyword,rep)

    return o_string


    

def upload_file(request):
    if request.method == 'POST':
        files=request.FILES.get("fileToUpload",None)
        with open(files.name, 'wb+') as destination:
            for chunk in files.chunks():
                destination.write(chunk)
        destination.close()
        filename=files.name
        if '.json' in filename:
            url='../result_json/'
        else :
            url='../result_xml/'

    return render(request, 'myapp/search.html', locals())

def search_result_xml(request):
    # if request.POST['keyword']=="Neoplasms" :
    #     #B="Neoplasms"
    #     request.POST['filename']="pubmed_hw4.xml"
    #     search_input1="cancer"

    #print(request.POST['filename'])
    #print(request.POST['filename'])

    print(request.POST['keyword'])
    #per_detail=ft_xml.process(request.POST['filename'],request.POST['keyword'],request.POST['keyword2'])
    #per_detail=ft_xml.process(request.POST['filename'],request.POST['keyword'])
    if request.POST['keyword']=="Neoplasms" :
        A="pubmed_hw4.xml"
        C="cancer"
    elif request.POST['keyword']=="Alzheimer Disease" :
        A="pubmed_hw4_az.xml"
        C="Alzheimer"
    elif request.POST['keyword']=="Influenza, Human" :
        A="pubmed_hw4_in.xml"
        C="influenza"
    per_detail=ft_xml.process(A,request.POST['keyword'],C)
    #edit_distance=ft_xml.minimumEditDistance(request.POST['keyword'],request.POST['keyword2'])
    #autocorrect_spall=ft_xml.autocorrect(request.POST['keyword'],request.POST['keyword2'])
    keyword1=request.POST['keyword']
    #keyword2=request.POST['keyword2']
    file_type='xml'


    return render(request, 'myapp/detail.html', locals())

def search_result_json(request):
    print(request.POST['filename'])
    print(request.POST['keyword'])
    #per_detail=ft_json.process(request.POST['filename'],request.POST['keyword'],request.POST['keyword2'])
    per_detail=ft_json.process(request.POST['filename'],request.POST['keyword'])
    #edit_distance=ft_xml.minimumEditDistance(request.POST['keyword'],request.POST['keyword2'])
    #autocorrect_spall=ft_xml.autocorrect(request.POST['keyword'],request.POST['keyword2'])
    keyword1=request.POST['keyword']
    #keyword2=request.POST['keyword2']
    file_type='json'
    return render(request, 'myapp/detail.html', locals())
