# SImple program to predict house prices based on house size

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation # animation support

# Lets generate some house sizes between 1000 and 3000 (typical sq footage)
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)

# Generate house prices from house size with a random noise added
np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_house)

# Plot the house and size
plt.plot(house_size, house_price, "bx") # bx is blue x
plt.ylabel("Price")
plt.xlabel("Size")
plt.show()

# The graph should be a more or less clustered graph and we should  be able to draw a line representing the model through the points

# Function to normalize the values to prevent under or overlfows
def nornmalize(array):
    return (array-array.mean())/array.std()

# Defined number of training samples, 0.7 = 70%. We can take the first 70% since the values are randomized
num_train_samples = math.floor(num_house * 0.7)

# Define the training data
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asanyarray(house_price[:num_train_samples:])

train_house_size_norm = nornmalize(train_house_size)
train_price_norm = nornmalize(train_price)

# Define the test data
test_house_size = np.asarray(house_size[num_train_samples:])
test_house_price = np.asanyarray(house_price[num_train_samples:])

test_house_size_norm = nornmalize(test_house_size)
test_house_price_norm = nornmalize(test_house_price)

# lets define some tensor placeholders. Placeholders are passed into the graph
tf_house_size = tf.placeholder("float", name="house_size")
tf_price = tf.placeholder("float", name="price")

# Define the variables holding the size_factor and the price we set during training
# We initialize them to some random values based on the normal distribution.
tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")

# Defining the operations for predicting values

tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)

tf_cost = tf.reduce_sum(tf.pow(tf_price_pred-tf_price, 2))/(2*num_train_samples)

learning_rate = 0.1

# Bacially we define the optimizer function that will minimize the loss defined in the operation "cost". We will use the gradient descent optimizer for this.
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

# Now we can initialize the variables and start a session

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Set some constants to display iterations
    display_every = 2
    num_training_iter = 50

    for iteration in range(num_training_iter):
        # Fit all training data on the graph
        for (x,y) in zip(train_house_size_norm, train_price_norm):
            sess.run(optimizer, feed_dict={tf_house_size: x, tf_price: y})
    
        # Print the current status
        if (iteration+1) % display_every ==0:
            c = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})
            print("iteration #:", '%04d' % (iteration+1), "cost=", "{:.9f}".format(c), \
            "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))

    print("Optimization completed!")
    training_cost = sess.run(tf_cost, feed_dict = {tf_house_size: train_house_size_norm, tf_price: train_price_norm})
    print("Trained cost=", training_cost, "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset), '\n')

    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()

    train_price_mean = train_price.mean()
    train_price_std = train_price.std()

    # Plot the graph
    plt.rcParams["figure.figsize"] = (10,8)
    plt.figure()
    plt.ylabel("Price")
    plt.xlabel("Size (sq.ft)")
    plt.plot(train_house_size, train_price, 'go', label='Training data')
    plt.plot(test_house_size, test_house_price, 'mo', label='Testing data')

    plt.plot(train_house_size_norm * train_house_size_std + train_house_size_mean, 
            (sess.run(tf_size_factor)*train_house_size_norm + sess.run(tf_price_offset)) * train_price_std + train_price_mean, label='Learned Regression')

    plt.legend(loc='upper left')
    plt.show()