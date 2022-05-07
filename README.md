# Digit classifier

## Contents

1. [Weekly report 1](./reports/week1.md)
2. [What is this project?](#what-is-this-project)
3. [UI and implementation](#ui-and-implementation)
4. [Sources](#sources)
5. [Running project locally](#running-project-locally)
6. [Course specific information](#course-specific-information)

### What is this project?

Following project was created for the algorithms and data structure course. The main functionality of the application will be to recognize user-written digits using convolutional neural network (CNN for short).

> NB! If there will be some time left, second model will be considered: SVM

### UI and implementation

The project will be implemented as a web app, using ReactJS for the frontend and Django-restframework as the backend. In the beginning as a scratchwork CNN will be implemented using scikit-learn. However, later it will be refactored and optimised from scratch.

### Sources

Main implementation of the CNN will be done with the help of **Andrew W. Trask's** _*Grokking Deep Learning*_ book. Further list of literature will be added throughout the implementation.

### Running project locally

> Nothing will work, if the data set is not present. I could not have left it in github repo, due to the size of files. Thus. you will have to download it yourself, via https://www.kaggle.com/datasets/oddrationale/mnist-in-csv , and put two of the files: mnist_test.csv and mnist_train.csv into neural_network directory.

In order to run the project locally you should install `Docker`. It is used to ensure that the project will actually run on the local machine due to containerizing the app and making sure that proper environments are used and required packages are installed.

After cloning repository, installing and starting docker enter the root directory of the project in shell (or terminal, depending on the OS) and run the following command

```
docker-compose up --build
```

In very simple terms, it will create containers, install all the required packages using specified package managers and OS. Then will run both the frontend and backend servers. If everything works properly you should see following output in the console:

```
tira_frontend | Compiled successfully!
tira_frontend |
tira_frontend | You can now view frontend in the browser.
tira_frontend |
tira_frontend |   Local:            http://localhost:3000
tira_frontend |   On Your Network:  http://172.18.0.3:3000
tira_frontend |
```

And following for the backend container:

```
tira_backend | System check identified no issues (0 silenced).
tira_backend | March 19, 2022 - 16:37:49
tira_backend | Django version 3.2.7, using settings 'backend.settings'
tira_backend | Starting development server at http://0.0.0.0:8000/
tira_backend | Quit the server with CONTROL-C.
```

Lastly, in order to set up backend properly ensure that you have the data set installed the way it is mentioned above. Then enter backend container by writing in another shell:

```
docker exec -it tira_backend bash
```

After that execute the following

```
python manage.py migrate
```

And lastly train the model:

```
python manage.py train_model 1000 300
```

This command will train the model using 1000 images and 300 iterations. Feel free to play around with the numbers, however be sure that the amount of images is not less that 500 as it is the hardcoded amount of batch size... Which should probably be changed to be a parameter as well

And you are ready to use application! Just use your favourite browser and enter:
`http://localhost:3000`

> NB! As it always happens with windows, some additional settings should be turned on: either Hyper-V or WSL(2) (preferably the latter one). But for more details check out docker installations docs online

### Course specific information

Degree: **BSc**

Language: **English**
