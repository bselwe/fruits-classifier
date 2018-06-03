import os
import math

from model import ModelBuilder
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint

train_dir = "/data/train"
valid_dir = "/data/validation"
output_dir = "/output"

img_width = 100
img_height = 100

num_classes = 9
batch_size = 32
num_epochs = 20

def main():
    num_train_samples = sum([len(files) for r, d, files in os.walk(train_dir)])
    num_valid_samples = sum([len(files) for r, d, files in os.walk(valid_dir)])

    num_train_steps = math.floor(num_train_samples / batch_size)
    num_valid_steps = math.floor(num_valid_samples / batch_size)

    model = ModelBuilder().build((img_width, img_height, 3), num_classes)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        zoom_range=[1, 1.15],
        rotation_range=15,
        width_shift_range=0.1,
        brightness_range=[0.8,1.2],
        fill_mode='wrap'
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    earlyStopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.1,
        patience=10,
        verbose=1,
        mode='auto')

    csvLogger = CSVLogger(
        filename=output_dir + '/training.log'
    )

    checkpointer = ModelCheckpoint(output_dir + '/model_checkpoint.h5', verbose=1, save_best_only=True)

    model.fit_generator(
        train_generator,
        steps_per_epoch=num_train_steps,
        epochs=num_epochs,
        validation_data=valid_generator,
        validation_steps=num_valid_steps,
        callbacks=[csvLogger, checkpointer]
    )

    model.save(output_dir + '/model_network.h5')


if __name__ == "__main__":
    main()