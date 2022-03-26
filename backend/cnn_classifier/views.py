from rest_framework.views import APIView
from rest_framework.response import Response
from PIL import Image

from .utils import get_image_from_data_url
from .models import UserDrawnImage
from .serializers import UserDrawnImageSerializer

# Views in drf are acting as middleman between server-client
# They are used to receive/ send/ update data, check permissions and etc.1
class UploadImageView(APIView):

  def post(self, request, format=None):
      print("---" * 10)
      base64_image = request.data.get('base64_image')
      image_file, other = get_image_from_data_url(base64_image)
      UserDrawnImage.objects.create(image=image_file)

      return Response('Response')
