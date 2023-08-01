import tensorflow as tf
from ingestion_module import DataIngestionTraining
import log_file as logging
from constant_variable import *
import os
from datetime import datetime
import logging


class TrainingConfig:
    def __init__(self):
        self.train_data = DataIngestionTraining().Train_DataGenerator()
        self.validation_data = DataIngestionTraining().Validation_DataGenerator()
        self.classification_labels = DataIngestionTraining().Label_Classification()

    def Model_Layer(self):
        try:
            logging.info("Initialising the model layering")
            logging.info("Downloading efficientnetB1 model_Weights including top and Base as false")
            base_model = tf.keras.applications.EfficientNetB1(include_top=False)
            base_model.trainable = True  # freeze base model layers
            logging.info("setting the top layer")
            inputs = tf.keras.layers.Input(shape=INPUT_SHAPE, name="input_layer")  # shape of input image
            logging.info("putting the base model in inference mode, so we can use it to extract features without updating the weights")
            x = base_model(inputs, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)  # pool the outputs of the base model
            logging.info("setting the Base layer")
            x = tf.keras.layers.Dense(len(self.classification_labels))(x)
            outputs = tf.keras.layers.Activation(FINAL_ACTIVATION, dtype=tf.float32, name="softmax_float32")(x)
            model = tf.keras.Model(inputs, outputs)
            model.summary()
            return model
        except Exception as e:
            raise Exception("error in layer function ", str(e))

    @staticmethod
    def callbacks():
        try:
            logging.info("initialising the callback functions")
            # Setting the Callback Function -
            # Create a function to implement a ModelCheckpoint callback with a specific filename

            timestamp = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
            saved_model_directory = os.path.join(SAVE_MODEL_PATH, timestamp)
            os.makedirs(saved_model_directory, exist_ok=True)
            filepath = f"{saved_model_directory}/{SAVED_MODEL_NAME}"

            savepointer = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, verbose=VERBOSE, save_best_only=True)

            # Create a function to implement an Early stop callback with loss monitor
            Early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=CALLBACK_PATIENCE,
                                                          verbose=VERBOSE)
            callbacks = [savepointer, Early_stop]
            return callbacks

        except Exception as e:
            raise Exception("error in callback function ", str(e))

    def model_compile_training(self):
        try:
            model_layers = TrainingConfig().Model_Layer()
            callbacks = TrainingConfig.callbacks()

            logging.info("compiling model")
            model_layers.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=LOSS, metrics=METRICS)
            logging.info("model compilation done")

            logging.info("initializing the model fitting/training")

            model_layers.fit(
                self.train_data,
                validation_data=self.validation_data,
                validation_steps=int(len(self.validation_data)),
                steps_per_epoch=int(len(self.train_data)),
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                callbacks=callbacks)
            logging.info("model with trainable bias saved")

        except Exception as e:
            raise ("error in model_compile_training function", e)

# print(TrainingConfig().model_compile_training())
