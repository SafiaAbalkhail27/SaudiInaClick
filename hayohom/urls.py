from django.urls import path
from . import views

urlpatterns = [
    path('', views.Index, name='Index'),
    path('ask', views.AskPage, name='AskPage'),
    path('Ask', views.Ask, name='Ask'),
    path('Record', views.Record, name='Record'),
]
