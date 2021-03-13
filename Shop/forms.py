from django import forms
from .models import ImageUploadModel,Post
from django.contrib.auth.forms import UserCreationForm
from django import forms
from django.contrib.auth.models import User



#  Now create customized user creation form like for adding email field in default django form.
class CreateUserform(UserCreationForm):

    # Meta class is simply inner class

    #add image field
    # image = forms.ImageField()
    class Meta:
        model = User
        fields = ['username','email','password1','password2']




class Rev(forms.ModelForm):
    class Meta:
        model = Post
        fields=['review']