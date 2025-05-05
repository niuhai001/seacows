"""
URL configuration for helloword project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
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

from django.urls import path,re_path
from . import views,testdb,search,search2
 
urlpatterns = [
    path('runoob/', views.runoob),
    path('testdb/', testdb.testdb),
    path('index/', views.index),
    path('hello/', views.hello),
    path('list/', views.list),
    path("list1/", views.list1),
    path("list2/", views.list2),
    path("list3/", views.list3),
    re_path(r'^search-form/$',search.search_form),
    re_path(r'^search/$', search.search),
    re_path(r'^post/$',search2.search_post),

]