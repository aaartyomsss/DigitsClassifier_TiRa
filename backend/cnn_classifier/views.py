from matplotlib import image
from rest_framework.views import APIView
from rest_framework.response import Response
from .utils import get_image_from_data_url
from .models import UserDrawnImage
from .serializers import UserDrawnImageSerializer
from .neural_network.neural_network import NeuralNetwork
from PIL import Image
from numpy import array

# Views in drf are acting as middleman between server-client
# They are used to receive/ send/ update data, check permissions and etc.1
class UploadImageView(APIView):

  def post(self, request, format=None):
      base64_image = request.data.get('base64_image')
      image_file, other = get_image_from_data_url(base64_image)
      (image_name, image_extension) = other
      # Saving image so that in the future data could be used to
      # train model even more

      UserDrawnImage.objects.create(image=image_file)

      img = Image.open(image_file).convert('L')
      # Numeric representation of the image
      data = array(img)
      # As our CNN was trained using this format - 
      # Input date should be formatted accordingly
      data = data.reshape(784, )

      cnn = NeuralNetwork()
      cnn = cnn.open_trained_model()

      ## Neural network accepts array of data and returns array of predictions
      result = cnn.predict([data])
      # Sending back the predicted result
      return Response({'result': result[0], 'image_name': image_name})


class RetrainNeuralNetwork(APIView):

    def post(self, request, format=None):
        # Mainly here for manual testing purposes for now.
        neural_network = NeuralNetwork()
        neural_network.train_model()
        neural_network.save_trained_model()
        return Response("Trained")