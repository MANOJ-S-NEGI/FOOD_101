import os
from constant_variable import *
import log_file as logging
import tensorflow
import logging


class DataIngestionTraining:
    def __init__(self):
        self.train_path = TRAIN_DIR
        self.validation_path = VALIDATION_DIR

    def Label_Classification(self):
        try:
            labels = os.listdir(self.train_path)
            return labels
        except Exception as e:
            raise Exception("error in Label_Classification ", str(e))

    def Train_DataGenerator(self):
        try:
            logging.info("creating Train data generator -get all the pixel values between 1 and 0")
            train_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(rescale=RESCALE,
                                                                                    shear_range=SHEAR_RANGE,
                                                                                    zoom_range=ZOOM_RANGE,
                                                                                    horizontal_flip=HORIZONTAL_FLIP,
                                                                                    vertical_flip=VERTICAL_FLIP)

            logging.info("created Train data generator -get all the pixel values between 1 and 0")
            logging.info("importing the data from dir and convert into the batching")
            train_data = train_datagen.flow_from_directory(self.train_path,
                                                           shuffle=SHUFFLE,
                                                           target_size=(TARGET_SIZE, TARGET_SIZE),
                                                           class_mode=CLASS_MODE,
                                                           batch_size=BATCH_SIZE,
                                                           )

            logging.info("Data into batches created as train_data and valid_data")
            return train_data

        except Exception as e:
            raise Exception("error in Train_Validation_DataGenerator ", str(e))

    def Validation_DataGenerator(self):
        try:
            logging.info("creating valid data generator -get all the pixel values between 1 and 0")
            validation_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(rescale=RESCALE,
                                                                                         shear_range=SHEAR_RANGE,
                                                                                         zoom_range=ZOOM_RANGE,
                                                                                         horizontal_flip=HORIZONTAL_FLIP,
                                                                                         vertical_flip=VERTICAL_FLIP)

            logging.info("created valid data generator -get all the pixel values between 1 and 0")

            logging.info("importing the data from dir and convert into the batching")

            valid_data = validation_datagen.flow_from_directory(self.validation_path,
                                                                shuffle=SHUFFLE,
                                                                target_size=(TARGET_SIZE, TARGET_SIZE),
                                                                class_mode=CLASS_MODE,
                                                                batch_size=BATCH_SIZE, )
            logging.info("Data into batches created valid_data")

            return valid_data

        except Exception as e:
            raise Exception("error in Train_Validation_DataGenerator ", str(e))


#print(DataIngestionTraining().Label_Classification())
#print(DataIngestionTraining().Train_DataGenerator())
#print(DataIngestionTraining().Validation_DataGenerator())
