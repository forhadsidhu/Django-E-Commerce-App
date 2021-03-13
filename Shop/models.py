from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class ImageUploadModel(models.Model):
    description = models.CharField(max_length=255,blank=True)
    document = models.ImageField(upload_to='image/%Y/%m/%d')
    uploaded_at = models.DateTimeField(auto_now_add=True)


# create  model for review submissions.
class Post(models.Model):
    review = models.TextField()



class Counter(models.Model):

    SURVEY_WIZARD_TYPE_CHOICES = (
                              ('SURVEY_WIZARD_ONE', 'survey_wizard_one'),
                              ('SURVEY_WIZARD_TWO', 'survey_wizard_two'),
                              ('SURVEY_WIZARD_THREE', 'survey_wizard_three'),
                              )

    survey_wizard_type = models.CharField(max_length=1000, choices=SURVEY_WIZARD_TYPE_CHOICES)
    pos_count = models.SmallIntegerField(default=0)
    neg_count = models.SmallIntegerField(default=0)
    total_count = models.SmallIntegerField(default=0)








