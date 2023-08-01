import os
import matplotlib.pyplot as plt
import tensorflow as tf
from random import random
from ingestion_module import DataIngestionTraining
from constant_variable import test_path

total_class_label = DataIngestionTraining().Label_Classification()
load_model = tf.keras.models.load_model("20230730-105918_food_101_model.hdf5")


def prediction_pipeline():
    # Make preds on a series of random images
    plt.figure(figsize=(20, 10))
    # setting true label:
    true_label_list = os.listdir(test_path)
    # selection random image folder:
    randum_selection_of_test_folder = random.choice(true_label_list)
    # setting the image dir:
    test_image_path = f"{test_path}/{randum_selection_of_test_folder}"
    # selecting the image randomly:
    image = random.choice(os.listdir(test_image_path))
    # full randomly selected image path of selected image
    filename = f"{test_image_path}/{image}"
    # Load the image and make predictions
    img = plt.imread(filename)
    img = tf.image.resize(img, [300, 300])
    img = img / 255
    image = tf.expand_dims(img, axis=0)
    pred_prob = load_model.predict(image)  # model accepts tensors of shape [None, 224, 224, 3]
    pred_class = total_class_label[pred_prob.argmax()]  # find the predicted class

    # Plot the image(s)
    plt.imshow(img)
    if randum_selection_of_test_folder == pred_class:  # Change the color of text based on whether prediction is right or wrong
        title_color = "g"
    else:
        title_color = "r"
    plt.title(f"actual: {randum_selection_of_test_folder}, \npred: {pred_class}, \nprob: {pred_prob.max():.2f}",
              c=title_color)
    plt.axis(False);
    plt.show()
