"""
It's recommended that this code be run in the Google Colab environment.

Comment/delete lines of importing drive and mounting drive if you are using on
local machine
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import time
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model


# Mounting Google Drive on Colab enviroment (only use this if you are on colab)
from google.colab import drive
drive.mount('/content/drive')

path_to_folder = '/content/drive/My Drive/yourfoldername'
test_dir = f'{path_to_folder}/dataset/test'
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

# Loading saved model
model_name = 'VGG16'
model = load_model(f'{path_to_folder}/Models/{model_name}')
evaluate = model.evaluate(test_generator, verbose=1)

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

# Transforming output prediction on int values
y_pred_categories = []
for prediction in y_pred:
    max_value = max(prediction)
    list_prediction = prediction.tolist()
    max_index = list_prediction.index(max_value)
    y_pred_categories.append(max_index)
print(y_pred_categories)


# Creating Confusion Matrix
cf_matrix = confusion_matrix(test_generator.classes, y_pred_categories)
print(cf_matrix)

# Creating a dataframe for Confusion Matrix
cm_df = pd.DataFrame(cf_matrix,
                     index=['Becks', 'Brahma', 'BudWeiser', 'Corona', 'Skol'],
                     columns=['Becks', 'Brahma', 'BudWeiser', 'Corona',
                              'Skol'])

# Plotting the Confusion Matrix
sns.heatmap(cm_df, annot=True, cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()


# Plotting the Percent Confusion Matrix
sns.heatmap(cm_df/np.sum(cm_df), annot=True, cmap='Blues', fmt='.2%')
plt.title('Percent Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()
