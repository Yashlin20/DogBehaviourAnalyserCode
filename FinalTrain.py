import tensorflow as tf
from keras.applications import efficientnet
from tensorflow.keras import models
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

image_size = 224
batch_size = 1
epochs = 40

# Method used to determine the number of steps for, during training, when the model iterates
# through the training photos and validation photos
def get_num_steps(num_samples, batch_size):
    if (num_samples % batch_size) > 0 :
        return (num_samples // batch_size) + 1
    else :
        return num_samples // batch_size


# Method used for transfer learning, where a pre-trained model (with layers frozen)
# is added to a model along with additional layers.
def transfer_learning(pre_trained_model):
    new_model = models.Sequential()
    new_model.add(pre_trained_model)
    pre_trained_model.trainable = False
    new_model.add(layers.GlobalAveragePooling2D())
    new_model.add(layers.Dense(2048, activation='relu'))
    new_model.add(layers.Dropout(rate = 0.2))
    new_model.add(layers.Dense(5,activation = 'softmax'))
    new_model.summary()
    return new_model

# Method used to generate training and validation data for the model to be trained and tested on respectively
def generate_data(image_size):

    # Methods of image augmentation applied to this image data generator
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                                                              rotation_range = 10, width_shift_range = 0.2,
                                                              height_shift_range = 0.2, zoom_range = 0.2,
                                                              horizontal_flip = True, fill_mode = 'nearest',vertical_flip = False)
    train_data = data_gen.flow_from_directory('PhotoDataset2/train', target_size=(image_size, image_size),
                                            batch_size= batch_size,
                                            class_mode='categorical',color_mode='rgb', seed = 42)
    validation_data = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory('PhotoDataset2/val', target_size = (image_size, image_size), batch_size = batch_size, class_mode = 'categorical', shuffle = False, color_mode='rgb')

    return train_data,validation_data

# Method used to train the overall CNN model
def train_CNN(CNN,train_data,validation_data,callbacks_list,num_train, num_validation):
    CNN.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(), metrics=['accuracy'])
    CNN.fit_generator(train_data, steps_per_epoch=get_num_steps(num_train, batch_size), epochs=epochs,
                              validation_data=validation_data, validation_steps=get_num_steps(num_validation, batch_size),
                              verbose=1, callbacks=callbacks_list)
    return CNN


# Defining pre-trained model used for transfer learning
pre_trained_model = efficientnet.EfficientNetB0(include_top= False, weights = 'imagenet')

# Variable used to store the number of training images
num_train = 0

image_list_train = os.listdir("PhotoDataset2/train")
for i in range(len(image_list_train)):
    image_list_sub_list_train = os.listdir("PhotoDataset2/train/" + image_list_train[i])
    for j in range(len(image_list_sub_list_train)):
        num_train = num_train + 1

# Variable used to store the number of validation
num_validation = 0

image_list_validation = os.listdir("PhotoDataset2/val")
for i in range(len(image_list_validation)):
    image_list_sub_list_validation = os.listdir("PhotoDataset2/val/" + image_list_validation[i])
    for j in range(len(image_list_sub_list_validation)):
        num_validation = num_validation + 1


CNN = transfer_learning(pre_trained_model)
CNN.summary()
data = generate_data(image_size)

# Callbacks list used for early stopping and reducing learning rate on plateu
callbacks_list = [
        EarlyStopping(
            monitor='val_loss',
            patience=2,
            mode='min',
            verbose=1

        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=1,
            min_lr=0.00001,
            verbose=1,
            mode='min'
        )]
trained_CNN = train_CNN(CNN, data[0], data[1], callbacks_list, num_train, num_validation)

# Saving of model after it has been trained
trained_CNN.save('model20-epoch46-batch1-B0AdamNoPP-UpdatedDataset7525Split')