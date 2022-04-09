# Following file will be used for global fixutes
# in order to avoid code repetition and use some
# setting in (almost) every test

import pytest
from rest_framework.test import APIClient

# APIClient will allow us to make requests 
# to the endpoints and thus to test them
@pytest.fixture
def api_client():
    return APIClient()