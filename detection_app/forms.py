# detection_app/forms.py

from django import forms

from detection_app.models import UploadedVideo

class ImageUploadForm(forms.Form):
    image = forms.ImageField()

class VideoUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedVideo
        fields = ['video']