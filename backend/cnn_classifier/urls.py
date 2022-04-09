from django.urls import path
from cnn_classifier.views import UploadImageView, RetrainNeuralNetwork

urlpatterns = [
  path('upload-image/', UploadImageView.as_view()),
  path('retrain-model/', RetrainNeuralNetwork.as_view())
]