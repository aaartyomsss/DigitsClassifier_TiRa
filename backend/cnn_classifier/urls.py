from django.urls import path
from cnn_classifier.views import UploadImageView

urlpatterns = [
  path('upload-image/', UploadImageView.as_view())
]