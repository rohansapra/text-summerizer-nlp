from django.urls import path, re_path
from . import views
from django.conf.urls import url
from .views import index

urlpatterns = [
    path('', views.index, name='index'),

    path('api/summerize',
         views.summerize_endpoint.as_view(), name="summerize_endpoint"),
]
