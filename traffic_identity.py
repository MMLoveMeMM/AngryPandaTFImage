# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 08:55:33 2018

@author: rd0348
"""

import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def load_data(data_dir):
    
    directories =[d for d in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir,d))]
    
    labels=[]
    images=[]
    for d in directories:
        label_dir=os.path.join(data_dir,d)
        file_names=[os.path.join(label_dir,f)
        for f in os.listdir(label_dir) if f.endswith(".ppm")]
        
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
            
    return images,labels

ROOT_PATH = "/BelgiumTS"
train_data_dir ="BelgiumTS/Training" # os.path.join(ROOT_PATH, "BelgiumTS/Training")
test_data_dir = "BelgiumTS/Testing" # os.path.join(ROOT_PATH, "BelgiumTS/Testing")

images, labels = load_data(train_data_dir)
print("Unique Labels: {0}\nTotal Images: {1}".format(len(set(labels)), len(images)))

def display_images_and_labels(images,labels):
    unique_labels=set(labels)
    plt.figure(figsize=(15,15))
    i=1
    for label in unique_labels:
        image=images[labels.index(label)]
        plt.subplot(8,8,i)
        plt.axis('off')
        plt.title("label{0} ({1})".format(label,labels.count(label)))
        i+=1
        _ = plt.imshow(image)
    plt.show()

# Resize images
images32 = [skimage.transform.resize(image, (32, 32))
                for image in images]
display_images_and_labels(images32,labels)

def display_label_images(images,label):
    limit=24
    plt.figure(figsize=(15,5))
    i=1
    
    start=labels.index(label)
    end=start+labels.count(label)
    for image in images[start:end][:limit]:
        plt.subplot(3,8,i)
        plt.axis('off')
        i+=1
        plt.imshow(image)
    plt.show()
    
display_label_images(images32,32)

def display_images_size(images):
    for image in images[:5]:
        print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))

display_images_size(images32)

# minimum viable model
labels_a = np.array(labels)
images_a = np.array(images32)
#print("labels: ", labels_a.shape, "\nimages: ", images_a.shape)

# Create a graph to hold the model.
graph = tf.Graph()

# Create model in the graph.
with graph.as_default():
    # Placeholders for inputs and labels.
    images_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])
    labels_ph = tf.placeholder(tf.int32, [None])

    # Flatten input from: [None, height, width, channels]
    # To: [None, height * width * channels] == [None, 3072]
    images_flat = tf.contrib.layers.flatten(images_ph)

    # Fully connected layer. 
    # Generates logits of size [None, 62]
    logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

    # Convert logits to label indexes (int).
    # Shape [None], which is a 1D vector of length == batch_size.
    predicted_labels = tf.argmax(logits, 1)

    # Define the loss function. 
    # Cross-entropy is a good choice for classification.
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph))

    # Create training op.
    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # And, finally, an initialization op to execute before training.
    # TODO: rename to tf.global_variables_initializer() on TF 0.12.
    init = tf.initialize_all_variables()

    print("images_flat: ", images_flat)
    print("logits: ", logits)
    print("loss: ", loss)
    print("predicted_labels: ", predicted_labels)


# Create a session to run the graph we created.
session = tf.Session(graph=graph)

# First step is always to initialize all variables. 
# We don't care about the return value, though. It's None.
_ = session.run([init])
for i in range(5001):
    _, loss_value = session.run([train, loss], 
                                feed_dict={images_ph: images_a, labels_ph: labels_a})
    if i % 10 == 0:
        print("Loss: ", loss_value)


# Pick 10 random images
sample_indexes = random.sample(range(len(images32)), 100)
sample_images = [images32[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

# Run the "predicted_labels" op.
predicted = session.run([predicted_labels], 
                        feed_dict={images_ph: sample_images})[0]
print(sample_labels)
print(predicted)

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
             fontsize=12, color=color)
    plt.imshow(sample_images[i])


# Load the test dataset.
test_images, test_labels = load_data(test_data_dir)


# Transform the images, just like we did with the training set.
test_images32 = [skimage.transform.resize(image, (32, 32))
                 for image in test_images]
display_images_and_labels(test_images32, test_labels)

# Run predictions against the full test set.
predicted = session.run([predicted_labels], 
                        feed_dict={images_ph: test_images32})[0]
# Calculate how many matches we got.
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
accuracy = match_count / len(test_labels)
print("Accuracy: {:.3f}".format(accuracy))
# Close the session. This will destroy the trained model.
session.close()











