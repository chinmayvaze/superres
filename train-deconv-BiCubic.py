import random
import glob
import subprocess
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
import wandb
from wandb.keras import WandbCallback
import pdb

run = wandb.init(project='superres')
config = run.config

config.num_epochs = 50
config.batch_size = 32
config.input_height = 256
config.input_width = 256
config.output_height = 256
config.output_width = 256

val_dir = 'data/test'
train_dir = 'data/train'

# automatically get the data if it doesn't exist
if not os.path.exists("data"):
    print("Downloading flower dataset...")
    subprocess.check_output(
        "mkdir data && curl https://storage.googleapis.com/wandb/flower-enhance.tar.gz | tar xz -C data", shell=True)

config.steps_per_epoch = len(
    glob.glob(train_dir + "/*-in.jpg")) // config.batch_size
config.val_steps_per_epoch = len(
    glob.glob(val_dir + "/*-in.jpg")) // config.batch_size


def image_generator_bicubic(batch_size, img_dir):
    """A generator that returns small images and large images.  DO NOT ALTER the validation set"""
    input_filenames = glob.glob(img_dir + "/*-in.jpg")
    counter = 0
    while True:
        intrp_images = np.zeros((batch_size, config.output_width, config.output_height, 3))
        large_images = np.zeros((batch_size, config.output_width, config.output_height, 3))
        random.shuffle(input_filenames)
        if counter+batch_size >= len(input_filenames):
            counter = 0
        for i in range(batch_size):
            img = input_filenames[counter + i]
            small_img = Image.open(img).convert("RGB")
            #pdb.set_trace()
            new_img = small_img.resize((config.output_width,config.output_height),Image.BICUBIC)
            #pdb.set_trace()
            intrp_images[i] = np.array(new_img) / 255.0
            large_images[i] = np.array(Image.open(img.replace("-in.jpg", "-out.jpg"))) / 255.0
        yield (intrp_images, large_images)
        counter += batch_size

      

def perceptual_distance(y_true, y_pred):
    """Calculate perceptual distance, DO NOT ALTER"""
    y_true *= 255
    y_pred *= 255
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))

def peak_snr (y_true, y_pred):
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]
    mse = K.mean(r*r + g*g + b*b)
    c_max = K.max(y_true)
    return (mse/(c_max*c_max))

def inception_block(ip):
    conv1x1_out = layers.Conv2D( 3, (1,1), padding='same', activation="relu")(ip)
    conv3x3_out = layers.Conv2D(10, (3,3), padding='same', activation="relu")(ip)
    conv5x5_out = layers.Conv2D(10, (5,5), padding='same', activation="relu")(ip)
    conv7x7_out = layers.Conv2D(10, (7,7), padding='same', activation="relu")(ip)
    concat_out  = layers.Concatenate()([conv1x1_out, conv3x3_out, conv5x5_out, conv7x7_out])
    #ups2x2_out  = layers.UpSampling2D((2,2))(concat_out)
    return concat_out

val_generator = image_generator_bicubic(config.batch_size, val_dir)
in_sample_images, out_sample_images = next(val_generator)


class ImageLogger(Callback):
    def on_epoch_end(self, epoch, logs):
        preds = self.model.predict(in_sample_images)
        in_resized = []
        for arr in in_sample_images:
            # Simple upsampling
            in_resized.append(arr.repeat(8, axis=0).repeat(8, axis=1))
        wandb.log({
            "examples": [wandb.Image(np.concatenate([in_resized[i] * 255, o * 255, out_sample_images[i] * 255], axis=1)) for i, o in enumerate(preds)]
        }, commit=False)



inp = layers.Input(shape=(config.output_width, config.output_height, 3))
inc1_out = inception_block (inp)
final_op = layers.Conv2D(3, (3,3), activation='relu', padding='same')(inc1_out)
model = Model(inp, final_op)

model.summary()
# DONT ALTER metrics=[perceptual_distance]
adam_opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)

model.compile(optimizer='adam', loss=[perceptual_distance],
              metrics=[perceptual_distance])

model.fit_generator(image_generator_bicubic(config.batch_size, train_dir),
                    steps_per_epoch=config.steps_per_epoch,
                    epochs=config.num_epochs, callbacks=[
                        ImageLogger(), WandbCallback()],
                    validation_steps=config.val_steps_per_epoch,
                    validation_data=val_generator)
