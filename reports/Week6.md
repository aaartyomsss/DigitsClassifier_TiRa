# What was done?

1. Added functionality for saving NN
2. Added mechanism to train NN in batches to improve the speed of training
3. Gradient decent documentation
4. Unit tests for functionality in point 1

# Progress?

This week main focus was documentation and researching the materials on how to improve the accuracy of the network without increasing drastically amount of training data. In addition to that, basic implementation of training NN in batches was added which improves speed of training at the expense of its accuracy.

# What did I learn?

I have understood better from maths POV what NN does while learning via gradient decent algorithm. I have also found out the way on how to improve its efficiency via usage of different activation functions such as softmax (at the output layer) and tanh (hidden layer). Those functions have been added to the model, however they are not implemented yet.

# What was problematic?

It is hard to optimise the network at this stage that there would have been a balance between the training time and its accuracy. Network perform quite poorly so far, even when it is being given 15000 training examples and 50-100 iterations - it does not score above 95 percent. Given more iterations and training examples - NN takes too much time to learn.

# Next?

Focus on the optimasation of the training speed and its accuracy should be the main priority for the next week.

# Total time spent this week

Approx: 10 hours
