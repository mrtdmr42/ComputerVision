# my first computer vision code

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np

src_path_train = "data/train/"
src_path_test = "data/test/"

# Define constants
image_width, image_height = 100, 100

# Define the augmentation and preprocessing configurations
train_datagen = ImageDataGenerator(
        rescale=1 / 255.0, # Normalize pixel values to range [0, 1]
        rotation_range=20, # Randomly rotate images in the range [-20, 20] degrees
        zoom_range=0.05, # Apply random zoom
        width_shift_range=0.05, # Randomly shift images horizontally by 5%
        height_shift_range=0.05, # Randomly shift images vetrically by 5%
        shear_range=0.05,  # Apply shear-based transformations
        horizontal_flip=True, # Randomly flip images horizontally
        fill_mode="nearest",
        validation_split=0.20)

test_datagen = ImageDataGenerator(rescale=1 / 255.0)

batch_size = 8
train_generator = train_datagen.flow_from_directory(
    directory=src_path_train,
    target_size=(100, 100),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    subset='training',
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    directory=src_path_test,
    target_size=(100, 100),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)

def prepare_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(100, 100, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(16, activation='relu')) # hidden layer has 16 neurons
    model.add(Dense(3, activation='sigmoid')) # output layer has 3 values: banana, apricot, cat
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model

model = prepare_model()
model.fit(train_generator,
                  epochs=10)

# Load and preprocess the test image
img_path = 'data/test/cat1.jpg'
img = image.load_img(img_path, target_size=(image_width, image_height))
img.show()
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

img_array /= 255.

# Make a prediction
prediction = model.predict(img_array)
print(prediction)
predicted_class = np.argmax(prediction)

# Print the predicted class
class_names = ['apricot', 'banana', 'cat']
print("Predicted class:", class_names[predicted_class])