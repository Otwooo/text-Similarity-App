from django.shortcuts import render, HttpResponse

def index(requese):
    return HttpResponse("Hello, world. You're at the polls index.")
