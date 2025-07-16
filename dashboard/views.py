from django.shortcuts import render



def home(request):
    return render(request, 'home.html')

def results(request):
    return render(request, 'results.html')

def prediction(request):
    return render(request, 'prediction.html')

