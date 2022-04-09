# Testing

At this stage the testing is done only on the backend side... And it is still rather simplistic :)

Testing is done with the help of `pytest` library that simplifies process a bit.

## Testing neural network

For the most part, current tests are rather trivial and are made with an idea in mind that they will simplify testing of the self-implemented CNN. The concerns are training speed, average accuracy and in the future - "unit accuracy", i.e. when we feed the network individual image and compare output with an actual result. It is not however a good practice, so the amount of those tests will be `very` limited.

## API testing

There is right now only 1 primary endpoint, which receives image, does the evaluation and then returns the result back to the frontend. There is a single test in `tests/test_api_calls.py` that is responsible for that.
In ordered to imitate the api calls - DRF's `APIClient` is used. It's instance is defined in `conftest.py` and then it is used in function as a parameter. This is one of the features of the `pytest` library, which reduces code duplication via the creation of `fixture`'s.

## Running tests

In order to run tests, first enter the container by writing:

```python
docker exec -it tira_backend bash
```

And then simply write:

```python
pytest
```

Note, that before doing so you will have to train the model using the frontend! Otherwise the tests will fail.

> At the time of writing this, I have understood that this could have been fixed with writing a management command and then running it either manually or before the tests, however this will be added in the future.
