"""
It's recommended that this code be run in the Google Colab environment.

Comment/delete lines of importing drive and mounting drive if you are using on
local machine
"""

from csv import writer
from os.path import exists
from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

# Mounting Google Drive on Colab enviroment (only use this if you are on colab)
from google.colab import drive
drive.mount('/content/drive')


model_name = 'VGG16'
epochs = 20
batch_size = 16
adam_lr = 0.000005

if model_name == 'VGG16':
    from keras.applications.vgg16 import VGG16, preprocess_input

path_to_folder = '/content/drive/My Drive/yourfoldername'
train_dir = f'{path_to_folder}/dataset/train'
validation_dir = f'{path_to_folder}/dataset/validation'

# Generating data and labels with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=7,
    width_shift_range=0.2,
    height_shift_range=0.2
)

valid_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

valid_generator = valid_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Load model
if model_name == 'VGG16':
    base_model = VGG16(weights="imagenet",
                       include_top=False,
                       pooling='max',
                       input_shape=(224, 224, 3))

# Allow training of the full network
for layer in base_model.layers:
    layer.trainable = True

# Adding Top Layers
x = base_model.output
x = Dense(1024, activation="relu")(x)
predictions = Dense(5, activation="softmax")(x)

# Creating the final model
model_final = Model(inputs=base_model.input, outputs=predictions)

# Compiling the model
model_final.compile(loss="categorical_crossentropy",
                    optimizer=tf.keras.optimizers.Adam(learning_rate=adam_lr),
                    metrics=["accuracy"])
model_final.summary()

# Stop with earlystopping
early = EarlyStopping(monitor='val_accuracy', patience=5,
                      restore_best_weights=True)


# Train the model
history = model_final.fit(
    train_generator,
    epochs=epochs,
    validation_data=valid_generator,
    callbacks=[early]
)

# Save Model
model_final.save(f'{path_to_folder}/Models/{model_name}')

# Appending log to a list
log_path = f'{path_to_folder}/log_training_model.csv'
file_exists = exists(log_path)
header_log = [
    'model_name',
    'batch_size',
    'adam_lr',
    'val_accuracy',
]

log_list = [
    model_name,
    batch_size,
    adam_lr,
    max(history.history['val_accuracy']),
]

# Create a file object for this file
with open(log_path, 'a') as f_object:
    writer_object = writer(f_object)
    # Add a header if the file not exists
    if not file_exists:
        writer_object.writerow(header_log)
    writer_object.writerow(log_list)
    f_object.close()
