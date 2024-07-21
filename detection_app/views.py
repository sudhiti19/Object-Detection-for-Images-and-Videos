# detection_app/views.py

import os
import sys
from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from django.core.files.storage import FileSystemStorage
from matplotlib import pyplot as plt
import numpy as np
from .models import UploadedImage, DetectionImage
from .forms import ImageUploadForm

sys.path.append(r'C:\Users\sanja\Desktop\practice\models\research\object_detection')

from detect_from_image import load_model, run_inference, load_image_into_numpy_array, run_inference_for_single_image
from object_detection.utils import label_map_util, visualization_utils as vis_util

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
    return render(request,'upload_video.html')