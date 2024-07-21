# detection_app/models.py

from django.db import models

class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

class DetectionImage(models.Model):
    original_image = models.ForeignKey(UploadedImage, on_delete=models.CASCADE)
    result_image = models.ImageField(upload_to='uploads/')
    detected_at = models.DateTimeField(auto_now_add=True)
