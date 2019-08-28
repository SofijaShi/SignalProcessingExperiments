#  raise URLError(err)
#urllib.error.URLError: <urlopen error [Errno 11001] getaddrinfo failed> problem with the URL because speech_data it's a file that fetches data from web with spoken numbers
from __future__ import division, print_function, absolute_import
import tflearn #library from the top of the tensorflow
import speech_data #file that fetches data from the web
import tensorflow as tf #google framework for machine learning

#hyperparametars
learning_rate = 0.0001 #higher learning_rate the faster our network trains, smaller- more accurate results
training_iters = 300000  # steps
batch_size = 64

width = 20  # mfcc features
height = 80  # (max) length of utterance
classes = 10  # digits

batch = word_batch = speech_data.mfcc_batch_generator(batch_size) #this function will download a set of wav files, each file is a recording of different spoken digit. Returns the labeled speach files as batch
X, Y = next(batch) # we are spliting the batch in training and testing data with next() function
trainX, trainY = X, Y # we are using the same data for testing, so it would be able to recognize the speaker i've trained on, but not other speakers
testX, testY = X, Y #overfit for now

# Network building
net = tflearn.input_data([None, width, height]) #gateway for the date to be put in the network, the parametar will help define the shape of the input  data
net = tflearn.lstm(net, 128, dropout=0.8) #building the next layer (number of neurons) too few - bad prediction, too many - overtraining
#dropout helps overfitting, by randomly turning off some of the neorons during training, so data is forced to find new paths in the network, alowing  more generalized model
net = tflearn.fully_connected(net, classes, activation='softmax') #conecting the neurons, number of classes is 10 because we are recognizing only 10 digits, activation will convert the data to probabilities
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy') #output layer that predicts the single number
# Training


###  "fix" for tensorflow version errors
col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for x in col:
    tf.add_to_collection(tf.GraphKeys.VARIABLES, x )


model = tflearn.DNN(net, tensorboard_verbose=0) #initalization of the network using DNN(deep neural netfork) built in function of tflearn
while 1: #training_iters
  model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True,
          batch_size=batch_size) #training and testing data with our specified batch_size
  _y=model.predict(X) # predict a spoken value from our training data
model.save("tflearn.lstm.model") #save the model for later use
print(model)
#print (_y)
#print (y)