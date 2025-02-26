# Generated by Django 4.2.14 on 2024-07-22 16:37

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('detection_app', '0004_remove_uploadedimage_detection_result_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='UploadedVideo',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('video', models.FileField(upload_to='uploads/')),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='DetectionVideo',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('result_video', models.FileField(upload_to='uploads/')),
                ('detected_at', models.DateTimeField(auto_now_add=True)),
                ('original_video', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='detection_app.uploadedvideo')),
            ],
        ),
    ]
