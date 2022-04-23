# What was done?

1. Minor fix to the UI
2. Initial work on the Native CNN (i.e. self-implemented)
3. Native CNN documentation

# Progress?

This week main focus was the implementation of the neural network from scratch, which has been successful rather successful as after testing it was possible to achieve accuracy greater that 90%.

The matrix operations are still performed with the help of numpy as:

a) It significantly improves code readability

b) Library is well-optimised and the operations are performed quickly

Lastly, I have started to document the implementation trying to explain in details, but with simple terms on how network operates.

> Better mathematical analysis will be provided later on.

# What did I learn?

For the most part I was revisiting the topics that I have already learned, but successfully forgot :) One of the key revision points were dealing with overfitting and avoidance of linearity.

# What was problematic?

At times dealing with matrix operations was confusing and coming up with the comparison of the actual output and the output produced by the neural network was not evident at first. The state of testing for now is also rather questionable and will require some improvements

# Next?

Additional functionality should be added to the neural network. It should not be so hardcoded as it is right now. User should choose himself the size of the hidden layer (preferably the number of hidden layers as well), alpha coefficient, activation function and other methods that should be used in CNN (which should also be added in a first place).

Documentation of the CNN should also be improved. Lastly, it would be nice to start working on the `"math behind CNN"` documentation.

# Total time spent this week

Approx: 14 hours
