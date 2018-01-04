# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 15:36:49 2018

@author: rd0348
使用TensorFlow 读取单个图片
"""

import tensorflow as tf
import matplotlib.image as mpimg  
import matplotlib.pyplot as plt  

reader = tf.WholeFileReader()
key, value = reader.read(tf.train.string_input_producer(['panda.jpg']))
image0 = tf.image.decode_jpeg(value)

image_resized = tf.image.resize_images(image0, [256, 256],
        method=tf.image.ResizeMethod.AREA)
plt.imshow(image_resized)
plt.show()

image_cropped = tf.image.crop_to_bounding_box(image0, 20, 20, 256, 256)

image_flipped = tf.image.flip_left_right(image0)

image_rotated = tf.image.rot90(image0, k=1)

image_grayed = tf.image.rgb_to_grayscale(image0)

img_resize_summary = tf.summary.image('image_resized', tf.expand_dims(image_resized, 0))
cropped_image_summary = tf.summary.image('image_cropped', tf.expand_dims(image_cropped, 0))
flipped_image_summary = tf.summary.image('image_flipped', tf.expand_dims(image_flipped, 0))
rotated_image_summary = tf.summary.image('image_rotated', tf.expand_dims(image_rotated, 0))
grayed_image_summary = tf.summary.image('image_grayed', tf.expand_dims(image_grayed, 0))
merged = tf.summary.merge_all()
print("process image ...")
#with tf.Session() as sess:
#  summary_writer = tf.summary.FileWriter('tmp/tensorboard', sess.graph)
#  summary_all = sess.run(merged)
#  summary_writer.add_summary(summary_all, 0)
#  summary_writer.close()

print("process image end !")


