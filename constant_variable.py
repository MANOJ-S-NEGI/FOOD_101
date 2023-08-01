TRAIN_DIR = "D:/FOOD_101/train"
VALIDATION_DIR = "D:/FOOD_101/validation"
test_path = "D:/FOOD_101/test/"
# generator module constants:

SHEAR_RANGE = 0.2
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True
RESCALE = 1 / 255
SHUFFLE = True
TARGET_SIZE = 244
CLASS_MODE = 'categorical'
BATCH_SIZE = 32
VERTICAL_FLIP = True

# TRAINING MODULE CONSTANTS:
KERNEL_SIZE = (3, 3)
PADDING = 'same'
INPUT_SHAPE = (244, 244, 3)
POOL_SIZE = (2, 2)
VERBOSE = 2
SAVE_MODEL_PATH = "saved_model_dir"
SAVED_MODEL_NAME = "food_101.keras"
CALLBACK_PATIENCE = 3
EPOCHS = 1
FINAL_ACTIVATION = "softmax"

LOSS = 'categorical_crossentropy'
METRICS = ["accuracy"]
