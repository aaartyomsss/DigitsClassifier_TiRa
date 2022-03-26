# Serializers are here to convert data from json to python format
# And vice versa (aka deserializers). Some validation and creation
# of the objects can be done with their help as well
from rest_framework.serializers import Serializer, ImageField
from .models import UserDrawnImage

class UserDrawnImageSerializer(Serializer):
    image = ImageField(required=True)

    def create(self, validated_data):
        print("THIS IS CALLED ! ! ! ! NO CAP")
        print("Validated data !~ ! ! ! ", validated_data)
        image = UserDrawnImage.objects.create(**validated_data)
        print(UserDrawnImage)

        # TODO: At this step ID should be returned by the client and
        # The network evaluation should be performed

        return image