from django.contrib import admin

from detection_app.models import DetectionImage, DetectionVideo, UploadedImage, UploadedVideo

# Register your models here.
class UploadedImageAdmin(admin.ModelAdmin):
    list_display = ('id', 'image')

class DetectionImageAdmin(admin.ModelAdmin):
    list_display = ('id', 'original_image', 'result_image')

admin.site.register(UploadedImage, UploadedImageAdmin)
admin.site.register(DetectionImage, DetectionImageAdmin)
class UploadedVideoAdmin(admin.ModelAdmin):
    list_display = ('id', 'video')

class DetectionVideoAdmin(admin.ModelAdmin):
    list_display = ('id', 'original_video', 'result_video')

admin.site.register(UploadedVideo, UploadedVideoAdmin)
admin.site.register(DetectionVideo, DetectionVideoAdmin)