import time
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model

# Change the path if you need
test_dir = "dataset/test"
batch_size = 16

# Generating data and labels with data augmentation
test_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=preprocess_input
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    classes=['Becks', 'Brahma', 'BudWeiser', 'Corona', 'Skol'],
    class_mode='categorical',
    shuffle=False
)

# Loading Model (put the path to your model folder)
model = load_model('path_to_your_saved_model')

# Inference test
runtimes = []
for i in range(11):
    start_time = time.time()
    y_pred = model.predict(test_generator, verbose=1)
    runtimes.append(time.time() - start_time)

# Removing the first element
runtimes = runtimes[1:]
avgRuntime = sum(runtimes) / len(runtimes)
print(avgRuntime)
print(runtimes)
