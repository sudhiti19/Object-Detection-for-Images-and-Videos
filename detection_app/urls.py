from django.urls import path
from .views import delete_detection_image, delete_detection_video, homepage, upload_image, upload_video, uploaded_files_list, view_image, view_video

urlpatterns = [
    path('', homepage, name='homepage'),
    path('upload/', upload_image, name='upload_image'),
    path('upload_video/', upload_video, name='upload_video'),
    path('view_video/<int:video_id>/', view_video, name='view_video'),
    path('view/<int:image_id>/', view_image, name='view_image'),
    path('uploaded_files/', uploaded_files_list, name='uploaded_files_list'),
    path('delete_detection_video/<int:pk>/', delete_detection_video, name='delete_detection_video'),
    path('delete_detection_image/<int:pk>/', delete_detection_image, name='delete_detection_image'),
]
