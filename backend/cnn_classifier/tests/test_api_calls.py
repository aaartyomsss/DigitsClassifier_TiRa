import pytest
import base64
import pathlib

# api_client is defined is conftest.py in root dir.
@pytest.mark.django_db
def test_endpoint_for_prediction(api_client):
    parent_directory_path = pathlib.Path(__file__).parent.resolve() # /tests
    path_to_image = f'{parent_directory_path}/test_images'
    image_name = 'test_image_seven.png'

    # Coverting image to same format as received from frontend
    with open(f'{path_to_image}/{image_name}', "rb") as img_file:
        # this is however in format of bytes, we need to proceed with
        # converting it to string.
        # Calling .decode() gets us the right representation
        b64_image = base64.b64encode(img_file.read()).decode('ascii')

        b64_image = f'data:image/png;base64,{b64_image}'

    # TODO: Add clean-up of the image added during the test
    
    data = {'base64_image': b64_image}
    print(b64_image)
    res = api_client.post('/api/upload-image/', data)

    # Successful operation
    assert res.status_code == 200
    # Actually correct prediction
    assert res.data == 7
