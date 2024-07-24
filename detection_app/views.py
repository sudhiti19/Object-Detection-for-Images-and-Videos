import os
import sys
import cv2
from django.conf import settings
from django.http import FileResponse, HttpResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.core.files.storage import FileSystemStorage
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import tensorflow as tf
from .models import DetectionVideo, UploadedImage, DetectionImage, UploadedVideo
from .forms import ImageUploadForm
import subprocess  # Add this import

sys.path.append(r'C:\Users\sanja\Desktop\practice\models\research\object_detection')

from detect_from_image import load_model, run_inference, load_image_into_numpy_array, run_inference_for_single_image
from object_detection.utils import label_map_util, visualization_utils as vis_util

# Patch the location of gfile
tf.gfile = tf.io.gfile

def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model

def homepage(request):
    return render(request, 'base.html')

def filter_output_dict(output_dict, category_index):
    """Filter out unwanted labels."""
    filtered_dict = {
        'detection_boxes': [],
        'detection_classes': [],
        'detection_scores': []
    }

    for i, label in enumerate(output_dict['detection_classes']):
        if label in category_index:
            filtered_dict['detection_boxes'].append(output_dict['detection_boxes'][i])
            filtered_dict['detection_classes'].append(label)
            filtered_dict['detection_scores'].append(output_dict['detection_scores'][i])

    # Convert lists to numpy arrays
    filtered_dict['detection_boxes'] = np.array(filtered_dict['detection_boxes'])
    filtered_dict['detection_classes'] = np.array(filtered_dict['detection_classes'])
    filtered_dict['detection_scores'] = np.array(filtered_dict['detection_scores'])

    return filtered_dict

def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        uploaded_file_url = fs.url(filename)

        input_image_path = os.path.join(settings.MEDIA_ROOT, filename)
        result_image_path = os.path.join(settings.MEDIA_ROOT, 'uploads', filename)

        labelmap_path = os.path.join(settings.BASE_DIR, 'detection_app', 'config', 'custom_label_map.pbtxt')
        model_path = 'C:\\Users\\sanja\\Desktop\\practice\\models\\research\\object_detection\\ssd_mobilenet_v2_320x320_coco17_tpu-8\\saved_model'
        detection_model = load_model(model_path)
        category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)

        image_np = load_image_into_numpy_array(input_image_path)
        output_dict = run_inference_for_single_image(detection_model, image_np)
        filtered_dict = filter_output_dict(output_dict, category_index)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            filtered_dict['detection_boxes'],
            filtered_dict['detection_classes'],
            filtered_dict['detection_scores'],
            category_index,
            instance_masks=filtered_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8
        )
        plt.figure(figsize=(4,4))

        plt.imshow(image_np)
        plt.savefig(result_image_path)

        # Save the UploadedImage instance
        uploaded_image = UploadedImage.objects.create(
            image=filename
        )

        # Save the DetectionImage instance
        detection_image = DetectionImage.objects.create(
            original_image=uploaded_image,
            result_image='uploads/' + filename
        )

        return redirect('view_image', image_id=detection_image.id)
    
    return render(request, 'upload_image.html')

def view_image(request, image_id):
    detection_image = get_object_or_404(DetectionImage, id=image_id)
    context = {
        'detection_image': detection_image,
    }
    return render(request, 'view_image.html', context)

def upload_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video = request.FILES['video']
        fs = FileSystemStorage()
        filename = fs.save(video.name, video)
        uploaded_file_url = fs.url(filename)

        input_video_path = os.path.join(settings.MEDIA_ROOT, filename)
        result_video_path = os.path.join(settings.MEDIA_ROOT, 'uploads', 'detected_' + os.path.splitext(filename)[0] + '.mp4')

        if not os.path.exists(os.path.join(settings.MEDIA_ROOT, 'uploads')):
            os.makedirs(os.path.join(settings.MEDIA_ROOT, 'uploads'))

        # Load model and category index
        labelmap_path = os.path.join(settings.BASE_DIR, 'detection_app', 'config', 'custom_label_map.pbtxt')
        model_path = 'C:\\Users\\sanja\\Desktop\\practice\\models\\research\\object_detection\\ssd_mobilenet_v2_320x320_coco17_tpu-8\\saved_model'
        detection_model = load_model(model_path)
        category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)

        # Open video capture and create video writer
        cap = cv2.VideoCapture(input_video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Resize dimensions
        new_width = width // 2
        new_height = height // 2
        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        out = cv2.VideoWriter(result_video_path, fourcc, fps, (new_width, new_height))

        # Process video frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame
            frame = cv2.resize(frame, (new_width, new_height))

            image_np = np.array(frame)
            output_dict = run_inference_for_single_image(detection_model, image_np)
            filtered_dict = filter_output_dict(output_dict, category_index)
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                filtered_dict['detection_boxes'],
                filtered_dict['detection_classes'],
                filtered_dict['detection_scores'],
                category_index,
                instance_masks=filtered_dict.get('detection_masks_reframed', None),
                use_normalized_coordinates=True,
                line_thickness=8
            )

            out.write(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Save video instances
        uploaded_video = UploadedVideo.objects.create(video=filename)
        detection_video = DetectionVideo.objects.create(
            original_video=uploaded_video,
            result_video='uploads/detected_' + os.path.splitext(filename)[0] + '.mp4'
        )

        # Play the processed video using the default media player
        video_player_path = "C:\\Program Files\\Windows Media Player\\wmplayer.exe"  # Path to Windows Media Player
        subprocess.run([video_player_path, result_video_path])

        return render(request, 'view_video.html', {'video_url': uploaded_file_url})

    return render(request, 'upload_video.html')

def view_video(request, video_id):
    detection_video = get_object_or_404(DetectionVideo, id=video_id)
    file_path = os.path.join(settings.MEDIA_ROOT, detection_video.result_video)
    response = FileResponse(open(file_path, 'rb'), content_type='video/mp4')
    response['Content-Disposition'] = f'inline; filename={os.path.basename(file_path)}'
    return response

def uploaded_files_list(request):
    detection_videos = DetectionVideo.objects.all()
    detection_images = DetectionImage.objects.all()
    context = {
        'detection_videos': detection_videos,
        'detection_images': detection_images
    }
    return render(request, 'uploaded_files_list.html', context)

def delete_detection_image(request, pk):
    if request.method == 'POST':
        image = get_object_or_404(DetectionImage, pk=pk)
        image.delete()
    return redirect('uploaded_files_list')  # Ensure this matches your URL pattern name

def delete_detection_video(request, pk):
    if request.method == 'POST':
        video = get_object_or_404(DetectionVideo, pk=pk)
        video.delete()
    return redirect('uploaded_files_list')
