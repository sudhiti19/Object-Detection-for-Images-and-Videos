from django.urls import path
from .views import homepage, upload_image, upload_video, view_image

urlpatterns = [
    path('', homepage, name='homepage'),
    path('upload/', upload_image, name='upload_image'),
    path('upload/video/',upload_video,name='upload_video'),
    path('view/<int:image_id>/', view_image, name='view_image'),
]
