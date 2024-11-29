#IMPORTING IMPORTING LIBRARIES
import numpy as np #IMPORTING NUMPY FOR CONVERTING IMAGES INTO MULTI_DIM ARRAYS
#IMPORTING PARTS OF TENSORFLOW AND KERAS FOR TRAINING A CNN MODEL FOR IMAGE CLASSIFICATION .
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam

#RESIZING IMAGE TO 128*128 PIXELS
img_width, img_height = 128, 128

#DEFINING BATCH SIZES FOR TRAINING THE MODEL ON DATA.
batch_size = 16

#CLASSIFYING DIFFERENT TYPES OF DATA I.E GREYSCALE AND RGB .
#DIVIDING THE TRAINING DATA INTO VALIDATION AND TESTING DATA.
train_datagen_greyscale = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_datagen_rgb = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

#RESIZING PIXEL VALUES FOR IMAGES TO THE RANGE OF(0-1) FOR GENERALIZATION WHICH WAS BEFORE IN SIZ(0-255)
validation_datagen_greyscale = ImageDataGenerator(rescale=1./255)
validation_datagen_rgb = ImageDataGenerator(rescale=1./255)

train_generator_greyscale = train_datagen_greyscale.flow_from_directory(
    'C:\\Users\\dell\\Downloads\\TURBO_BINARY\\data\\train\\greyscale data',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale')

train_generator_rgb = train_datagen_rgb.flow_from_directory(
    'C:\\Users\\dell\\Downloads\\TURBO_BINARY\\data\\train\\rgb data',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    color_mode='rgb')

#COMBINING DATA FOR RANDOM TESTING.
def combined_generator(generator_greyscale, generator_rgb):
    while True:
        batch_greyscale = next(generator_greyscale)
        batch_rgb = next(generator_rgb)
        batch_greyscale_images = np.concatenate([batch_greyscale[0]] * 3, axis=-1)
        combined_images = np.concatenate((batch_greyscale_images, batch_rgb[0]), axis=0)
        combined_labels = np.concatenate((batch_greyscale[1], batch_rgb[1]), axis=0)
        
        yield combined_images, combined_labels


#GENERATING ARRAYS FOR  VALIDATION DATA.
validation_generator_greyscale = validation_datagen_greyscale.flow_from_directory(
    'C:\\Users\\dell\\Downloads\\TURBO_BINARY\\data\\train\\greyscale data',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale')

validation_generator_rgb = validation_datagen_rgb.flow_from_directory(
    'C:\\Users\\dell\\Downloads\\TURBO_BINARY\\data\\train\\validation\\rgb data',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    color_mode='rgb')


def combined_validation_generator(generator_greyscale, generator_rgb):
    while True:
        batch_greyscale = next(generator_greyscale)
        batch_rgb = next(generator_rgb)
        batch_greyscale_images = np.concatenate([batch_greyscale[0]] * 3, axis=-1)
        combined_images = np.concatenate((batch_greyscale_images, batch_rgb[0]), axis=0)
        combined_labels = np.concatenate((batch_greyscale[1], batch_rgb[1]), axis=0)
        yield combined_images, combined_labels



#DEFINING LAYERS FOR CNN MODEL
num_channels = 3
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, num_channels)))  #THE CORE LAYER FOR CNN
model.add(Activation('relu')) #TO SHORTLIST REQUIRED AND SPECIAL FEATURES.
model.add(MaxPooling2D(pool_size=(2, 2))) #IT TAKES MAX VALUE OF THE MATRIX OF EACH PIXEL TO DOWNSIZE THE DATA.
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) #CONVERTS 2D DATA INTO 1D DATA.
model.add(Dense(64)) #CONNECTS ALL THE LAYERS OF THE NEURAL NETWORK.
model.add(Activation('relu'))
model.add(Dropout(0.5)) #REMOVES REDUNDANT PART OF DATA.
model.add(Dense(1)) #
model.add(Activation('sigmoid')) #COMBIMES THE RESULT DATA.
model.compile(optimizer=Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy']) #ADAM IS USED TO OPTIMISE THE MODEL FOR BETTER ACCURACY.

train_generator = combined_generator(train_generator_greyscale, train_generator_rgb) #START TRAINING
validation_generator = combined_validation_generator(validation_generator_greyscale, validation_generator_rgb) #START VALIDATION.
steps_per_epoch = (train_generator_greyscale.samples + train_generator_rgb.samples)
validation_steps = (validation_generator_greyscale.samples + validation_generator_rgb.samples)


#FITTING THE TRAINED MODEL FOR THE DATA.
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_steps)

#EVALUATING THE MODEL.
model.evaluate(validation_generator, steps=validation_steps)

#SAVING THE FINAL MODEL
model.save("C:\\Users\\dell\\Downloads\\TURBO_BINARY\\data\\train\\trained_model.h5")

