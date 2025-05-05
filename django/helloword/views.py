from django.shortcuts import render
 
def runoob(request):
    views_str = "菜鸟教程"
    return render(request, "runoob.html", {"views_str": views_str})

def index(request):
    vars = "<a href='https://www.runoob.com/django/django-tutorial.html'>菜鸟教程</a>"
    return render(request,"runoob.html" ,{"vars":vars})

def hello(request):
    num = 80
    return render(request, "runoob.html", {"num": num})

def list(request):
    my_list = ["apple", "banana", "orange"]
    return render(request, "runoob.html", {"my_list": my_list})

def list1(request):
    views_dict = {"name":"菜鸟教程","age":18}
    return render(request, "runoob.html", {"views_dict": views_dict})
    
def list2(request):
     views_list = ["a", "b", "c", "d", "e"]
     return render(request, "runoob.html", {"listvar": views_list})

def list3(request):
    name ="菜鸟教程"
    return render(request, "runoob.html", {"name": name})


