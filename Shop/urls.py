from django.urls import path
from . import views

urlpatterns = [
    path('',views.home, name = 'home'),
    path('templates/',views.face, name = 'uimage'),
    path('templates/login',views.login, name = 'log'),
    path('logout/', views.logoutUser, name='logout'),
    path('templates/',views.registerPage, name = 'reg'),
    path('templates/reviews',views.prodRev, name = 'rev_submit')



]


