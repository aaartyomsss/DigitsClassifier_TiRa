# What was done?

1. Convolutional layer added
2. Removed Scikit from the project
3. Added functionality to avoid overfitting by performing preliminary stoppage
4. Native CNN is now used in the API
5. Add management command to train the model

# Progress?

This week implementation of the most important part was performed, i.e. the convolutional layer. Convolutional layer should improve the accuracy of the CNN, however, will increase the time required to train the network in the first place as there are drastically more matrix operations and transformations that are performed.
In addition to that management command that allows to train the model via shell/ cmd/ terminal was added.

# What did I learn?

I have now understood the pros and cons of the convolutional layer in CNN and how effective it is when it comes to accuracy, but also how computationaly heavy the method is on its own. The signle iteration on my own local machine with the amount of images = 1000 and batch_size of 200 has taken 24 seconds.

# What was problematic?

Yet again, finding correct parameters was quite a daunting task. Also, training model on the large set of data has taken large amount of time. Thus, for convenience sake the end model that will be used in the demo will be trained on a rather limited data. But, if there are certain enthusiasts who would like to check if the model performs better on the whole data set - then I would love to see the outcome myself

# Next?

Before the demo I would love to clean up the code and improve documentation. An lastly, actually train the model on the relatively big amount of data and save it.

# Total time spent this week

Approx: 13 hours
