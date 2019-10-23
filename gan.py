import keras
from keras.layers import Conv2D, UpSampling2D, Input, BatchNormalization, Embedding, Reshape, Dense, Flatten, Concatenate, Conv2DTranspose, Dropout
from keras.models import Model, load_model
from keras.initializers import RandomNormal
from keras import optimizers
from keras import losses
import text_encoding
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import random
import re
import skimage.io as io
from skimage.transform import resize
from scipy.stats import truncnorm
import os

#################### Common ####################
vectors_file_path = "pre-trained_GloVe/glove.6B/glove.6B.300d.txt"
annotations_file_path = "COCO/annotations_trainval2017/annotations/captions_train2017.json"
images_dir_path = "COCO/train2017/"
output_dir = "generated_images/"
words_to_remove = ["a", "an", "the"]
annotations_per_image = 5
epochs = 1000
batch_size = 32
d_label_smooting = 0.3
image_input_shape = (32, 32, 3, )
image_output_shape = (32, 32, 3, )
text_input_shape = (16, 300, 1, )
lrelu = lambda x: keras.layers.LeakyReLU(alpha=0.2)(x)
g_optimizer = optimizers.Adam(lr=0.0002, beta_1=0.5)
d_optimizer = optimizers.Adam(lr=0.0002, beta_1=0.5)
fake = np.zeros((batch_size, 1))
real = np.ones((batch_size, 1))
condition_input, condition_output = text_encoding.encode_text(lrelu, text_input_shape)
initial_weights = RandomNormal(mean=0.0, stddev=0.02)
#g_loss = losses.binary_crossentropy()
#d_loss = losses.binary_crossentropy()
################################################
# Generator

def generator():
    noise_input_g = Input(shape=image_input_shape)

    network_g = Conv2D(16, (5, 5), activation="relu", kernel_initializer=initial_weights, padding='same')(noise_input_g)
    network_g = BatchNormalization()(network_g)
    network_g = Conv2D(16, (5, 5), activation="relu", kernel_initializer=initial_weights, padding='same')(noise_input_g)
    network_g = BatchNormalization()(network_g)
    network_g = Conv2D(32, (5, 5), activation="relu", kernel_initializer=initial_weights, padding='same')(network_g)
    network_g = BatchNormalization()(network_g)
    network_g = Conv2D(32, (5, 5), activation="relu", kernel_initializer=initial_weights, padding='same')(network_g)
    network_g = BatchNormalization()(network_g)
    
    concatenated_input_g = Concatenate()([network_g, condition_output])
    # size 32x32x35

    # encoder
    network_g = Conv2D(64, (3, 3), activation="relu", kernel_initializer=initial_weights, padding='same')(concatenated_input_g)
    network_g = BatchNormalization()(network_g)
    #network_g = BatchNormalization()(network_g)
    network_g = Dropout(0.5)(network_g)
    network_g = Conv2D(64, (3, 3), activation="relu", strides=(2, 2), kernel_initializer=initial_weights, padding='same')(network_g)
    network_g = BatchNormalization()(network_g)
    # size 16x16x64

    network_g = Conv2D(128, (3, 3), activation="relu", kernel_initializer=initial_weights, padding='same')(network_g)
    network_g = BatchNormalization()(network_g)
    network_g = Conv2D(128, (3, 3), activation="relu", strides=(2, 2), kernel_initializer=initial_weights, padding='same')(network_g)
    network_g = BatchNormalization()(network_g)
    # size 8x8x128

    network_g = Conv2D(256, (3, 3), activation="relu", kernel_initializer=initial_weights, padding='same')(network_g)
    network_g = BatchNormalization()(network_g)
    network_g = Conv2D(256, (3, 3), activation="relu", strides=(2, 2), kernel_initializer=initial_weights, padding='same')(network_g)
    network_g = BatchNormalization()(network_g)
    # size 4x4x256
    network_g = Dropout(0.5)(network_g)
    network_g = Conv2D(512, (3, 3), activation="relu", kernel_initializer=initial_weights, padding='same')(network_g)
    network_g = BatchNormalization()(network_g)
    network_g = Conv2DTranspose(512, (3, 3), activation="relu", strides=(2, 2), kernel_initializer=initial_weights, padding='same')(network_g)
    network_g = BatchNormalization()(network_g)
    # size 8x8x512

    network_g = Conv2D(256, (3, 3), activation="relu", kernel_initializer=initial_weights, padding='same')(network_g)
    network_g = BatchNormalization()(network_g)
    network_g = Conv2DTranspose(256, (3, 3), activation="relu", strides=(2, 2), kernel_initializer=initial_weights, padding='same')(network_g)
    network_g = BatchNormalization()(network_g)
    # size 16x16x256

    network_g = Conv2D(128, (3, 3), activation="relu", kernel_initializer=initial_weights, padding='same')(network_g)
    network_g = BatchNormalization()(network_g)
    network_g = Conv2DTranspose(128, (3, 3), activation="relu", strides=(2, 2), kernel_initializer=initial_weights, padding='same')(network_g)
    network_g = BatchNormalization()(network_g)
    # size 32x32x128
    network_g = Dropout(0.5)(network_g)
    #network_g = BatchNormalization()(network_g)
    network_g = Conv2D(64, (5, 5), activation="relu", kernel_initializer=initial_weights, padding='same')(network_g)
    network_g = BatchNormalization()(network_g)
    network_g = Conv2D(32, (5, 5), activation="relu", kernel_initializer=initial_weights, padding='same')(network_g)
    network_g = BatchNormalization()(network_g)
    network_g = Conv2D(16, (5, 5), activation="relu", kernel_initializer=initial_weights, padding='same')(network_g)
    network_g = BatchNormalization()(network_g)
    network_output_g = Conv2D(3, (5, 5), activation="tanh", kernel_initializer=initial_weights, padding='same')(network_g)
    # first output will be image 32x32x3

    model_g = Model(inputs=[noise_input_g, condition_input], outputs=network_output_g)
    return model_g

def discriminator():
    input_d = Input(shape=image_output_shape)

    network_d = Conv2D(16, (5, 5), activation=lrelu, kernel_initializer=initial_weights, padding='same')(input_d)
    network_d = BatchNormalization()(network_d)
    network_d = Conv2D(32, (5, 5), activation=lrelu, kernel_initializer=initial_weights, padding='same')(network_d)
    network_d = BatchNormalization()(network_d)
    network_d = Dropout(0.5)(network_d)
    network_d = Conv2D(32, (5, 5), activation=lrelu, kernel_initializer=initial_weights, padding='same')(network_d)
    network_d = BatchNormalization()(network_d)

    concatenated_input_d = Concatenate()([network_d, condition_output])
    # size 32x32x35

    network_d = Conv2D(64, (5, 5), activation=lrelu, kernel_initializer=initial_weights, padding='same')(concatenated_input_d)
    network_d = BatchNormalization()(network_d)
    network_d = Dropout(0.5)(network_d)
    network_d = Conv2D(64, (3, 3), activation=lrelu, strides=(2, 2), kernel_initializer=initial_weights, padding='same')(network_d)
    network_d = BatchNormalization()(network_d)
    # size 16x16x64

    network_d = Conv2D(128, (3, 3), activation=lrelu, kernel_initializer=initial_weights, padding='same')(network_d)
    network_d = Dropout(0.5)(network_d)
    network_d = BatchNormalization()(network_d)
    network_d = Conv2D(256, (3, 3), activation=lrelu, strides=(2, 2), kernel_initializer=initial_weights, padding='same')(network_d)
    network_d = BatchNormalization()(network_d)
    # size 8x8x128
    network_d = Conv2D(256, (3, 3), activation=lrelu, kernel_initializer=initial_weights, padding='same')(network_d)
    network_d = Conv2D(128, (3, 3), activation=lrelu, strides=(2, 2), kernel_initializer=initial_weights, padding='same')(network_d)
    network_d = BatchNormalization()(network_d)
    # size 4x4x128
    network_d = Conv2D(64, (3, 3), activation=lrelu, kernel_initializer=initial_weights, padding='same')(network_d)
    network_d = Dropout(0.5)(network_d)
    network_d = BatchNormalization()(network_d)
    network_d = Conv2D(32, (3, 3), activation=lrelu, kernel_initializer=initial_weights, padding='same')(network_d)
    network_d = BatchNormalization()(network_d)
    network_d = Flatten()(network_d)
    network_output_d = Dense(1, activation="sigmoid", kernel_initializer=initial_weights)(network_d)

    model_d = Model(inputs=[input_d, condition_input], outputs=network_output_d)
    model_d.compile(optimizer=d_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model_d

def create_input_noise(input_shape=(32, 32, 3), batch=1):
    values_count = input_shape[0] * input_shape[1] * input_shape[2] * batch
    return np.random.randn(values_count).reshape((batch, ) + input_shape)

def load_vector_representations(file_path=""):
    vector_representations = {}
    lines = []
    with open(file_path, "r", encoding="utf8") as f:
        lines = f.readlines()

    line = lines.pop()

    while line:
        splitted_line = line.split()
        key = splitted_line[0].lower()
        if key not in words_to_remove and key.isalpha() and key not in vector_representations:
            #vector_representations[key] = [-1.0 if float(feature) < -1  else 1.0 if float(feature) > 1 else float(feature) for feature in splitted_line[1:]]
            vector_representations[key] = [float(feature) for feature in splitted_line[1:]]
        try:
            line = lines.pop()
        except IndexError:
            break
        
    return vector_representations

def parse_annotation(annotation=""):
    annotation = annotation.lower()
    annotation = re.sub(r"[^a-z ]", r"", annotation)
    for word in words_to_remove:
        annotation = re.sub("(?<= ){}(?![a-z])".format(word), r"", annotation)
    for word in words_to_remove:
        annotation = re.sub("(?<![a-z]){}(?= )".format(word), r"", annotation)
    annotation = re.sub(r" {2}", r" ", annotation)
    nr_char_to_remove = 0
    while not annotation[nr_char_to_remove].isalpha():
        nr_char_to_remove += 1
    if nr_char_to_remove != 0:
        annotation = annotation[nr_char_to_remove:]
        nr_char_to_remove = 0
    while not annotation[nr_char_to_remove - 1].isalpha():
        nr_char_to_remove -= 1
    if nr_char_to_remove != 0:
        annotation = annotation[:nr_char_to_remove]
    return annotation

def create_annotation_embedding(vector_representations=None, annotation=""):
    annotation_embedding = []
    for word in annotation.split(" "):
        try:
            annotation_embedding.append(vector_representations[word])
        except KeyError:
            continue
    annotation_embedding = np.array(annotation_embedding)
    annotation_embedding = np.resize(annotation_embedding, text_input_shape)
    return annotation_embedding

def load_image_ids(coco=None):
    image_ids = coco.getImgIds()
    return image_ids

def load_image_annotations(coco=None, image_ids=[]):
    annotation_ids = coco.getAnnIds(imgIds=image_ids)
    annotations = coco.loadAnns(annotation_ids)
    parsed_annotations = []
    parsed_annotation = []
    annotation_counter = 0
    for annotation in annotations:
        annotation_counter += 1
        parsed_annotation.append(parse_annotation(annotation["caption"]))
        if annotation_counter == annotations_per_image:
            parsed_annotations.append(parsed_annotation)
            parsed_annotation = []
            annotation_counter = 0
    return parsed_annotations

def load_images(coco=None, image_ids=[]):
    images_data = coco.loadImgs(image_ids)
    images = [io.imread(images_dir_path + image_data['file_name']).astype("float32") for image_data in images_data]
    images = np.array(images)
    images = (images - 127.5) / 127.5
    images = [resize(image, image_output_shape, anti_aliasing=True) for image in images] 
    images = np.array(images)
    return images

def get_annotations_embbeddings(coco=None, vector_representations=None, image_ids=[]):
    images_annotations = load_image_annotations(coco, image_ids)
    annotations_embbeddings = []
    annotations = []
    for image_annotations in images_annotations:
        annotation_index = random.randint(0, annotations_per_image - 1)
        annotations.append(image_annotations[annotation_index])
        annotation_embbedding = create_annotation_embedding(vector_representations, image_annotations[annotation_index])
        annotations_embbeddings.append(annotation_embbedding)
    return np.array(annotations_embbeddings), annotations

def plot_generated_images(epoch=1, generator=None, coco=None, vector_representations=None, image_ids=[]):
    noise = create_input_noise(image_input_shape, batch_size)
    chosen_image_ids = random.sample(image_ids, batch_size)
    annotations_embbeddings, annotations = get_annotations_embbeddings(coco, vector_representations, chosen_image_ids)
    generated_images = generator.predict([noise, annotations_embbeddings])
    i = 0
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for generated_image, annotation in zip(generated_images, annotations):
        plt.figure(figsize=(10,10))
        plt.subplot(4, 4, 1)
        plt.imshow((generated_image * 127.5 + 127.5).astype(int), interpolation='nearest')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir + 'image_{}_{}.png'.format(epoch, annotation))
        i += 1
    plt.close("all")


def train(generator=None, discriminator=None, gan=None, epochs=1000, batch_size=1):
    vector_representations = load_vector_representations(vectors_file_path)
    coco = COCO(annotations_file_path)
    image_ids = coco.getImgIds()
    samples_nr = int((len(image_ids) / batch_size) / 5)
    #real = np.ones((batch_size, 1))
    #generated = np.zeros((batch_size, 1))

    #One-sided label smoothing
    real_d = np.ones((batch_size, 1)) - random.uniform(0, d_label_smooting)
    real_g = np.ones((batch_size, 1))
    generated = np.zeros((batch_size, 1)) + random.uniform(0, d_label_smooting)

    for epoch in range(epochs):
        tmp_image_ids = image_ids[:]
        for sample_nr in range(samples_nr):
            chosen_image_ids = random.sample(tmp_image_ids, batch_size)
            
            images = load_images(coco, chosen_image_ids)
            input_noise = create_input_noise(image_input_shape, batch_size)

            annotations_embbeddings, _ = get_annotations_embbeddings(coco, vector_representations, chosen_image_ids)

            generated_images = generator.predict([input_noise, annotations_embbeddings])
            discriminator_loss_real = 0
            discriminator_loss_generated = 0
            flip_labels = (random.randint(0, 50) == 13)
            if flip_labels:
                discriminator_loss_real, _ = discriminator.train_on_batch([images, annotations_embbeddings], generated)
                discriminator_loss_generated, _ = discriminator.train_on_batch([generated_images, annotations_embbeddings], real_d)
            else:
                discriminator_loss_real, _ = discriminator.train_on_batch([images, annotations_embbeddings], real_d)
                discriminator_loss_generated, _ = discriminator.train_on_batch([generated_images, annotations_embbeddings], generated)

            for chosen_image_id in chosen_image_ids:
                tmp_image_ids.remove(chosen_image_id)

            chosen_image_ids = random.sample(tmp_image_ids, batch_size)
            annotations_embbeddings, _ = get_annotations_embbeddings(coco, vector_representations, chosen_image_ids)

            generator_loss = gan.train_on_batch([input_noise, annotations_embbeddings], real_g)

            print ("epoch = %d, sample_nr = %d of %d, d_r = %.3f, d_g = %.3f g = %.3f, fliped = %r" \
                % (epoch, sample_nr, samples_nr, discriminator_loss_real, discriminator_loss_generated, generator_loss, flip_labels))
            
        discriminator.save_weights('discriminator_weights.h5')
        gan.save_weights('gan_weights.h5')
        generator.save_weights('generator_weights.h5')
        plot_generated_images(epoch, generator=generator, coco=coco, vector_representations=vector_representations, image_ids=image_ids)


if __name__ == "__main__":
    model_g = generator()
    model_d = discriminator()
    noise_input_g = Input(shape=image_input_shape)

    out_g = model_g([noise_input_g, condition_input])
    model_d.trainable = False
    out_d = model_d([out_g, condition_input])

    model_gan = Model(inputs=[noise_input_g, condition_input], outputs=out_d)
    model_gan.compile(loss='binary_crossentropy', optimizer=g_optimizer)
    #model_gan.summary()

    train(generator=model_g, discriminator=model_d, gan=model_gan, epochs=epochs, batch_size=batch_size)
    model_d.save_weights('discriminator_weights.h5')
    model_gan.save_weights('gan_weights.h5')
    model_g.save_weights('generator_weights.h5')