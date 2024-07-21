from django.contrib import admin

from detection_app.models import DetectionImage, UploadedImage

# Register your models here.
class UploadedImageAdmin(admin.ModelAdmin):
    list_display = ('id', 'image')

class DetectionImageAdmin(admin.ModelAdmin):
    list_display = ('id', 'original_image', 'result_image')

admin.site.register(UploadedImage, UploadedImageAdmin)
admin.site.register(DetectionImage, DetectionImageAdmin)