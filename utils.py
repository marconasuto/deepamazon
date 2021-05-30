#!/usr/bin/env python
# coding: utf-8

# In[535]:


import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_io as tfio
from tensorflow import keras
import elasticdeform.tf as etf #https://github.com/gvtulder/elasticdeform
import cv2
from spectral import *
import skimage
from skimage import io
import tempfile
from PIL import Image
import datetime 
import time
import os
import io
import shutil
import random
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import py7zr
import tarfile
import pandas as pd 
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.style as style
import matplotlib.image as mpimg
import seaborn as sns
from tqdm import tqdm
from collections import Counter
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


# In[2]:


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


# In[1026]:


def params(epochs,
           train_val_test,
           ref_means,
           ref_stds,
        image_size=128,
           color_mode='rgb',
           endswith='.jpg',
          channels=3,
          use_random_flip=False,
          use_90rotation=False,
          use_elastic_transform=False,
          horizontal_flip=False,
           vertical_flip=False,
           rotation_range=False,
           shear_range=False,
          batch_size=128,
           learning_rate=1e-4,
           momentum=0.9,
           num_classes=17,
           loss_type='binary_crossentropy',
           opt_type='Adam',
           tags=['agriculture', 'artisinal_mine','bare_ground','blooming','blow_down','clear','cloudy','conventional_mine','cultivation','habitation','haze','partly_cloudy','primary','road','selective_logging','slash_burn','water'],
           columns=['agriculture', 'artisinal_mine','bare_ground','blooming','blow_down','clear','cloudy','conventional_mine','cultivation','habitation','haze','partly_cloudy','primary','road','selective_logging','slash_burn','water'],
           vegetation_index=False,
           rlronplateau=True,
           checkpoint=True,
           trainable='Full',
           pcg_unfreeze=0,
           preprocess=True,
          regularization=False):

        

    train_params = AttrDict({'size': image_size,
                             'channels': channels,
                             'color_mode': color_mode,
                             'endswith':endswith,
                             'use_random_flip': use_random_flip,
                             'use_90rotation': use_90rotation,
                             'use_elastic_transform': use_elastic_transform,
                             'horizontal_flip': horizontal_flip,
                             'vertical_flip': vertical_flip,
                             'rotation_range': rotation_range,
                             'shear_range': shear_range,
                             'batch_size': batch_size,
                             'num_epochs': epochs,
                             'learning_rate': learning_rate,
                             'momentum': momentum,
                             'num_classes': num_classes,
                             'num_samples':len(train_val_test[0]),
                             'seed': np.random.seed(123),
                             'loss_type':loss_type,
                             'opt_type': opt_type,
                             'loss_obj': loss_obj(loss_type),
                             'optimizer_obj': optimizer(learning_rate, momentum, opt_type),
                             'tags': tags,
                             'split_train_test':0.9,
                             'split_train_val':0.7,
                             'columns': columns,
                             'num_images_train': len(train_val_test[0]),
                             'num_images_val': len(train_val_test[1]),
                             'num_images_test': len(train_val_test[-1]),
                            'vegetation_index':vegetation_index,
                            'rlronplateau':rlronplateau,
                            'checkpoint':checkpoint,
                            'trainable':trainable,
                            'pcg_unfreeze':pcg_unfreeze,
                            'preprocess':preprocess,
                             'regularization':regularization,
                            'ref_means':ref_means,
                            'ref_stds':ref_stds})

    test_params = AttrDict({'size': image_size,
                            'batch_size': len(train_val_test[-1]),
                            'color_mode':color_mode,
                            'endswith':endswith,
                            'channels':channels,
                            'preprocess':preprocess,
                            'seed': np.random.seed(123)})
    return train_params, test_params


# In[48]:


def make_cooccurence_matrix(df, labels, title):
    labels_df = df[labels] 
    path = './reports/assets/'
    co_oc = labels_df.T.dot(labels_df)
    plt.figure(dpi=200)
    sns.heatmap(co_oc, cmap='mako_r')
    title_text = 'Co-occurency matrix: {}'.format(title)
    plt.title(title_text, size=18, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(path, title_text+'.png'))
    return co_oc


# In[7]:


def calibrate_image(rgb_image, params):
    # Transform test image to 32-bit floats to avoid 
    # surprises when doing arithmetic with it 
    calibrated_img = rgb_image.copy().astype('float32')

    # Loop over RGB
    for i in range(3):
        # Subtract mean 
        calibrated_img[:,:,i] = calibrated_img[:,:,i]-np.mean(calibrated_img[:,:,i])
        # Normalize variance
        calibrated_img[:,:,i] = calibrated_img[:,:,i]/np.std(calibrated_img[:,:,i])
        # Scale to reference 
        calibrated_img[:,:,i] = calibrated_img[:,:,i]*params.ref_stds[i] + params.ref_means[i]
        # Clip any values going out of the valid range
        calibrated_img[:,:,i] = np.clip(calibrated_img[:,:,i],0,255)

    # Convert to 8-bit unsigned int
    return calibrated_img.astype('uint8')


# In[138]:


def reference_calibration_values(set_filenames):
    # Image intensity calibration
    copy_set= set_filenames.copy()
    np.random.shuffle(copy_set)
    random_list = copy_set[:100]
    ref_colors = [[],[],[]]
    for _file in random_list:
        # keep only the first 3 bands, RGB
        _img = mpimg.imread(_file)[:,:,:3]
        # Flatten 2-D to 1-D
        _data = _img.reshape((-1,3))
        # Dump pixel values to aggregation buckets
        for i in range(3): 
            ref_colors[i] = ref_colors[i] + _data[:,i].tolist()

    ref_colors = np.array(ref_colors)
    ref_means = [np.mean(ref_colors[i]) for i in range(3)]
    ref_stds = [np.std(ref_colors[i]) for i in range(3)]
    return ref_means, ref_stds


# In[9]:


#functions for plotting cmap 
def cmap_rescale(elements):
    result = []
    if isinstance(elements, dict):
        _max = max(elements.values())
        _min = min(elements.values())
        result = [(el - _min) / (_max - _min) for el in elements.values()]
    if isinstance(elements, list):
        _max = np.max(elements.values())
        _min = np.min(elements.values())
        result = [(el - _min) / (_max - _min) for el in elements.values()]
    return result

def percentage(count_tags):
    _sum = sum(count_tags.values())
    return [ (el/_sum)*100 for el in count_tags.values()] 


# In[110]:


def load_data_using_keras(folders, df, data_filenames_img, params):
    image_generator = {}
    data_generator = {}
    
    for _dir, _filenames in zip(folders, data_filenames_img):
        end = _dir.split('/')[-1]

        if params.preprocess:
            if end == 'train':
                image_generator[end] = ImageDataGenerator(horizontal_flip=params.horizontal_flip,
                                              vertical_flip=params.vertical_flip,
                                             rotation_range=params.rotation_range,
                                             shear_range=params.shear_range)
                
                data_generator[end] = image_generator[end].flow_from_dataframe(
                    dataframe = df[df['filename'].isin(_filenames)],
                    x_col = 'filename',
                    y_col = params.columns,
                    batch_size=params.batch_size,
                    directory=_dir,
                    seed = params.seed,
                    shuffle=True,
                    target_size=(params.size, params.size),
                    class_mode='raw',
                    color_mode=params.color_mode)

            if end == 'val':
                image_generator[end] = ImageDataGenerator()
                
                data_generator[end] = image_generator[end].flow_from_dataframe(
                    dataframe = df[df['filename'].isin(_filenames)],
                    x_col = 'filename',
                    y_col = params.columns,
                    batch_size=params.batch_size,
                    directory=_dir,
                    seed = params.seed,
                    shuffle=False,
                    target_size=(params.size, params.size),
                    class_mode='raw',
                    color_mode=params.color_mode)
            if end == 'test':
                image_generator[end] = ImageDataGenerator()
                
                data_generator[end] = image_generator[end].flow_from_dataframe(
                    dataframe = df[df['filename'].isin(_filenames)],
                    x_col = 'filename',
                    y_col = params.columns,
                    batch_size=len(df[df['filename'].isin(_filenames)]),
                    directory=_dir,
                    seed = params.seed,
                    shuffle=False,
                    target_size=(params.size, params.size),
                    class_mode='raw',
                    color_mode=params.color_mode)
                
        else:
            if end == 'train':
                image_generator[end] = ImageDataGenerator(horizontal_flip=params.horizontal_flip,
                                              vertical_flip=params.vertical_flip,
                                             rotation_range=params.rotation_range,
                                             shear_range=params.shear_range,
                                                         rescale=1./255)
                
                data_generator[end] = image_generator[end].flow_from_dataframe(
                    dataframe = df[df['filename'].isin(_filenames)],
                    x_col = 'filename',
                    y_col = params.columns,
                    batch_size=params.batch_size,
                    directory=_dir,
                    seed = params.seed,
                    shuffle=True,
                    target_size=(params.size, params.size),
                    class_mode='raw',
                    color_mode=params.color_mode)

            if end == 'val':
                image_generator[end] = ImageDataGenerator(rescale=1./255)
                data_generator[end] = image_generator[end].flow_from_dataframe(
                    dataframe = df[df['filename'].isin(_filenames)],
                    x_col = 'filename',
                    y_col = params.columns,
                    batch_size=params.batch_size,
                    directory=_dir,
                    seed = params.seed,
                    shuffle=False,
                    target_size=(params.size, params.size),
                    class_mode='raw',
                    color_mode=params.color_mode)
            if end == 'test':
                image_generator[end] = ImageDataGenerator(rescale=1./255)
                data_generator[end] = image_generator[end].flow_from_dataframe(
                    dataframe = df[df['filename'].isin(_filenames)],
                    x_col = 'filename',
                    y_col = params.columns,
                    batch_size=len(df[params.split_train_test:]),
                    directory=_dir,
                    seed = params.seed,
                    shuffle=False,
                    target_size=(params.size, params.size),
                    class_mode='raw',
                    color_mode=params.color_mode)

    return data_generator


# In[12]:


def mkdir(path):
    new_dir = path
    if not os.path.exists(path):
        os.mkdir(path)


# In[14]:


def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    
    buf = io.BytesIO()
    
    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format='png')
    
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    
    # Use tf.image.decode_png to convert the PNG buffer
    # to a TF image. Make sure you use 4 channels.
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    
    # Use tf.expand_dims to add the batch dimension
    image = tf.expand_dims(image, 0)
    
    return image


# In[15]:


def plot_confusion_matrix(cm, columns):
    fig = plt.figure(figsize=(10,20))
    for i, (label, matrix) in enumerate(zip(columns, cm)):
        ax = plt.subplot(6,3,i+1)
        labels = [f'not_{label}', label]
        sns.heatmap(matrix, 
                    ax=ax, 
                    annot = True, 
                    square = True, 
                    fmt = '.0f', 
                    cbar = False, 
                    cmap = 'Blues', 
                    xticklabels = labels, 
                    yticklabels = labels, 
                    linecolor = 'black', 
                    linewidth = 1)
        plt.title(labels[1], size=8)
        plt.subplots_adjust(wspace=5, hspace=5)
        ax.set_yticklabels(labels, va='center', position=(0,0.28), size=8)
        ax.set_xticklabels(labels, ha='center', position=(0.28,0), size=8)
        plt.xlabel('PREDICTED CLASS', labelpad=10)
        plt.ylabel('TRUE CLASS', labelpad=10)
        plt.tight_layout()
        
    return fig


# In[502]:


def elastic_transform(image):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """ 
    displacement_val = np.random.randn(2, 3, 3) * 5
    displacement = tf.Variable(displacement_val)

    elastic = etf.deform_grid(image, displacement, axis=(0,1))

    return elastic


# In[17]:


def loss_obj(loss_type):
    if loss_type == 'binary_crossentropy':
        loss_obj = tf.keras.losses.BinaryCrossentropy()
    if loss_type == 'focal_loss':
        loss_obj = tfa.losses.SigmoidFocalCrossEntropy()
    if loss_type == 'soft_f2':
        loss_obj = micro_soft_f2
    return loss_obj


# In[18]:


def optimizer(learning_rate, momentum, opt_type):
    if opt_type == 'SGD_momentum':
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    if opt_type == 'Adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate, decay=0.0001)
    return opt


# In[485]:


def parse_image(filename, label, params):
    '''# This function parses jpeg or tiff images from images path, returns tensors of unscaled/rescaled and resized images'''
    
    if params.endswith == '.jpg':
        if params.preprocess:
            parse_image = tf.io.read_file(filename)
            decoded_image = tf.io.decode_jpeg(parse_image)
            image_resized = tf.image.resize(decoded_image, [params.size, params.size])
        else:
            parse_image = tf.io.read_file(filename)
            decoded_image = tf.image.convert_image_dtype(tf.io.decode_jpeg(parse_image), tf.float32)
            image_resized = tf.image.resize(decoded_image, [params.size, params.size])
            
    elif params.endswith == '.tif':
        if params.preprocess:
            parse_image = tf.io.read_file(filename)
            decode_image = tfio.experimental.image.decode_tiff(parse_image)
            # extract the rgb values
            bgr_image = decode_image[:,:,:3]
            rgb_image = bgr_image[:, :, [2,1,0]]
            calibrated_image = calibrate_image(rgb_image, params.ref_means, params.ref_stds)
            image_resized = tf.image.resize(calibrated_image, [params.size, params.size]) # resize to input size (i.e. from 256x256 to 128x128)
        else:
            parse_image = tf.io.read_file(filename)
            decode_image = tfio.experimental.image.decode_tiff(parse_image)
            # extract the rgb values
            bgr_image = decode_image[:,:,:3]
            rgb_image = bgr_image[:, :, [2,1,0]]
            image_rescaled = rgb_image / 255
            calibrated_image = calibrate_image(image_rescaled, params.ref_means, params.ref_stds)
            image_resized = tf.image.resize(calibrated_image, [params.size, params.size]) # resize to input size (i.e. from 256x256 to 128x128)
            
    return image_resized[:,:,:params.channels], label


# In[455]:


def preprocess(image, label, params):
    """Image preprocessing for training.
        You can apply the following operations:
        - Horizontally flip the image with probability 1/2
        - Apply 90 deg rotation
        - Apply elastic transformation
    """
    if params.use_random_flip:
        image = tf.image.random_flip_left_right(image)
    
    if params.use_90rotation:
        image = tf.image.rot90(image)
    
    if params.use_elastic_transform:
        image = elastic_transform(image)
            
    # Make sure the image is still in [0, 1]
    if params.preprocess == False:
        image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


# In[298]:


def create_filenames_labels(df, 
                            endswith='.jpg', 
                      pcg_dataset=1, 
                      train_data_dir ='./data/train',
                      val_data_dir='./data/val',
                      test_data_dir='./data/test',
                      raw_data_dir='./data/raw/train/jpg/train-jpg copy'):
    # pcg_dataset = percentage of total files to use: i.e. 30% of 40479 samples = 12143 samples
    # empty data dirs
    data_dirs = [train_data_dir, val_data_dir, test_data_dir]
    for data_dir in data_dirs:
        for file in os.listdir(data_dir):
            if file != 'vegetation_index':
                os.remove(os.path.join(data_dir, file))
    # create lists of filenames for train, val, test sets
    # copy lists of images from raw folder to train, val, test folders using lists of filenames 
    
    pcg_total_files = int(pcg_dataset * len(df))

    filenames_raw = os.listdir(raw_data_dir)
    filenames = [os.path.join(raw_data_dir, f) for f in filenames_raw if f.endswith(endswith)]

    seed = random.seed(123)
    filenames.sort()
    random.shuffle(filenames) # files come from same distribution, hence random shuffling won't negatively affect test, train and val distributions

    filenames = filenames[:pcg_total_files]


    split_train_test = int(0.9 * len(filenames)) # 10% for testing, 90% for val and train
    train_filenames_raw = filenames[:split_train_test]
    test_filenames_raw = filenames[split_train_test:]

    split_train_val = int(0.7 * len(train_filenames_raw)) # 80% for train, 20% for val 
    val_filenames_raw = train_filenames_raw[split_train_val:]
    train_filenames_raw = train_filenames_raw[:split_train_val]


    train_val_test = [train_filenames_raw, val_filenames_raw, test_filenames_raw]
    dest_dirs = [train_data_dir, val_data_dir, test_data_dir]

    for filename_dir, dest_dir in tqdm(zip(train_val_test, dest_dirs)):
        if len(os.listdir(dest_dir)) != len(filename_dir): #check if directory is empty
            for filename in filename_dir:
                shutil.copy(filename, dest_dir)
                
    # get lists of filenames with new path (i.e. '.data/train/img_name.jpg')

    train_filenames = []
    val_filenames = []
    test_filenames = []

    for filename_dir, dest_dir in tqdm(zip(train_val_test, dest_dirs)):
        for filename in filename_dir:
            if dest_dir == train_data_dir:
                train_filenames.append(os.path.join(dest_dir, filename.split('/')[-1]))
            elif dest_dir == val_data_dir:
                val_filenames.append(os.path.join(dest_dir, filename.split('/')[-1]))
            elif dest_dir == test_data_dir:
                test_filenames.append(os.path.join(dest_dir, filename.split('/')[-1]))
                
    # MultiLabelBinarizer to one hot encode labels
    mlb = MultiLabelBinarizer()
    df['ohe-tags'] = [el for i,el in zip(df.index, mlb.fit_transform(df.tags.map(lambda x: x.split(' '))))]
    
    # create lists of labels one hot encoded
    train_val_test = [train_filenames, val_filenames, test_filenames]
    train_labels = []
    val_labels = []
    test_labels = []

    train_val_test_labels = [train_labels, val_labels, test_labels]

    for _filenames, _labels in zip(train_val_test, train_val_test_labels):
        for _filename in _filenames:
            _labels.append(np.asarray(df[df.image_name == _filename.split('/')[-1].split(endswith)[0]]['ohe-tags'].iloc[0]))
    
    #get names of images for each set
    train_filenames_img = [el.split('/')[-1] for el in train_filenames_raw]
    val_filenames_img = [el.split('/')[-1] for el in val_filenames_raw]
    test_filenames_img = [el.split('/')[-1] for el in test_filenames_raw]

    data_filenames_img = [train_filenames_img, val_filenames_img, test_filenames_img]
    
    print('Total number of samples (train + val + test) (%d %% of original dataset) : %d' %(pcg_dataset*100, len(filenames)))
    print('Training set - number of samples: %d' % len(train_filenames_raw))
    print('Validation set - number of samples: %d' % len(val_filenames_raw))
    print('Test set - number of samples: %d' % len(test_filenames_raw))

    print('Training set - number of samples in .data/train: %d' % len(os.listdir(train_data_dir)))
    print('Validation set - number of samples .data/val: %d' % len(os.listdir(val_data_dir)))
    print('Test set - number of samples .data/test: %d' % len(os.listdir(test_data_dir)))
    
    return train_val_test, train_val_test_labels, data_filenames_img
    


# In[108]:


def create_dataset(is_training, filenames, labels, params):
    """"Create the input data pipeline using `tf.data` instead of ImageDataGenerator as it is 5x faster.

    The filenames have format "train_{}.jpg.
    For instance: "data_dir/train_4584.jpg".

    Args:
        is_training: (bool) whether to use the train or test pipeline.
                     At training, we shuffle the data and have multiple epochs
        filenames: (list) filenames of the images, as ["data/train/train_{}.tif"...]
        labels: (list) corresponding list of labels
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    num_samples = len(filenames)
    assert len(filenames) == len(labels), "Filenames and labels should have same length"

    # Create a Dataset serving batches of images and labels
    # We don't repeat for multiple epochs because we always train and evaluate for one epoch
    parse_fn = lambda f, l: parse_image(f, l, params)
    train_fn = lambda f, l: preprocess(f, l, params)

    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(tf.cast(labels, tf.float32))))
            .shuffle(num_samples) # shuffling before mapping for performance reasons as explained in https://stackoverflow.com/questions/51909997/tensorflow-dataset-shuffle-before-map-map-and-batch  
            .map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .map(train_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(params.batch_size)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  #tf.data.experimental.AUTOTUNE will prompt the tf.data runtime to tune the value dynamically at runtime.
        )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.cast(labels, tf.float32)))
            .map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(params.batch_size)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # make sure you always have one batch ready to serve
        )

    # Create reinitializable iterator from dataset
    return dataset


# In[23]:


def vegetation_index(image):
    '''This function returns a 6 channels image (b, g, r, nir, ndvi, gndvi) given a 4 channels image (b, g, r, nir)'''
    r = image[:,:,2]
    nir = image[:,:,3]
    g = image[:,:,1]
    ndvi = (nir - r) / (nir + r)
    gndvi = (nir - g) / (nir + g)
    image_stacked = tf.stack([image[:,:,0], 
                              image[:,:,1],
                              image[:,:,2],
                              image[:,:,3],
                              tf.clip_by_value(ndvi, 0.0, 1.0),
                              tf.clip_by_value(gndvi, 0.0, 1.0)], axis=-1)
    return image_stacked


# In[24]:


def parse_image_with_vi(filename, params):
    '''This function parses images and calculates vegetation indices (ndvi, ngdvi) stacking ndvi and ngdvi as 2 channels to the original 4 channels images; from (b, g, r, nir) -> (b, r, g, nir, ndvi, ngdvi)'''
    image_string = tf.io.read_file(filename) #read string
    image_decoded = tfio.experimental.image.decode_tiff(image_string) # decode string to img array with p channels (4)
    image_rescaled = tf.image.convert_image_dtype(image_decoded, tf.float32) # rescale 1/255 and cast float
    image_resized = tf.image.resize(image_rescaled, [params.size, params.size]) # resize to input size (i.e. from 256x256 to 128x128)
    image_resized = vegetation_index(image_resized) # add ndvi and gndvi as 2 channels (b, g, r, nir, ndvi, gndvi)
    return image_resized


# In[25]:


def write_vi_TFRdataset(save_path, dataset_name, input_set, params, image=True):
    '''This function writes a .TFRecord dataset from tensors (i.e., images, labels), applying vegetation index (multichannel images)'''
    if image:
        # Write images to file
        parse_fn = lambda f: parse_image_with_vi(f, params)
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(input_set))).map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).map(tf.io.serialize_tensor)
        writer = tf.data.experimental.TFRecordWriter(os.path.join(save_path, dataset_name + '.tfrecord'))
        writer.write(dataset)
    else:
        # Write labels to file
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(tf.cast(input_set, tf.float32)))).map(tf.io.serialize_tensor)
        writer = tf.data.experimental.TFRecordWriter(os.path.join(save_path, dataset_name + '.tfrecord'))
        writer.write(dataset)

def read_TFRdataset(load_path, dataset_name):
    # Read from file
    parse_tensor_f32 = lambda x: tf.io.parse_tensor(x, tf.float32)
    dataset = (tf.data.TFRecordDataset(os.path.join(load_path, dataset_name + '.tfrecord')).map(parse_tensor_f32))
    return dataset


# In[26]:


def create_model(version, params):
    
    if version == 'v1.0':
        # Baseline
        inputs = Input(shape=(params.size, params.size, params.channels))
        x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = Conv2D(32, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        
        outputs = Dense(params.num_classes, activation='sigmoid')(x)  
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(loss=params.loss_obj, optimizer=params.optimizer_obj, metrics=[f2_score])
        
        
    if version == 'v1.1':
        # v1.0 with 128 units in FC layer w.r.t 64 
        inputs = Input(shape=(params.size, params.size, params.channels))
        x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = Conv2D(32, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(64, 3, activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = Conv2D(64, 3, activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        
        outputs = Dense(params.num_classes, activation='sigmoid')(x)  
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(loss=params.loss_obj, optimizer=params.optimizer_obj, metrics=[f2_score])
    
    if version == 'v1.2':
        # v1.3 with dropout layers after each block
        inputs = Input(shape=(params.size, params.size, params.channels))
        x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = Conv2D(32, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.2)(x)
        
        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.2)(x)
        
        x = Conv2D(128, 3, activation='relu', padding='same')(x)
        x = Conv2D(128, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.2)(x)
        
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        outputs = Dense(params.num_classes, activation='sigmoid')(x)  
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(loss=params.loss_obj, optimizer=params.optimizer_obj, metrics=[f2_score])
    
    if version == 'v1.3':
        inputs = Input(shape=(params.size, params.size, params.channels))
        x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = Conv2D(32, 3, activation='relu', padding='same')(x)
        x = Conv2D(32, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.2)(x)

        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.2)(x)
        
        x = Conv2D(128, 3, activation='relu', padding='same')(x)
        x = Conv2D(128, 3, activation='relu', padding='same')(x)
        x = Conv2D(128, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.2)(x)
        
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        outputs = Dense(params.num_classes, activation='sigmoid')(x)  
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(loss=params.loss_obj, optimizer=params.optimizer_obj, metrics=[f2_score])
        
    if version == 'v1.4':
        inputs = Input(shape=(params.size, params.size, params.channels))
        x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = Conv2D(32, 3, activation='relu', padding='same')(x)
        x = Conv2D(32, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.2)(x)

        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.2)(x)
        
        x = Conv2D(128, 3, activation='relu', padding='same')(x)
        x = Conv2D(128, 3, activation='relu', padding='same')(x)
        x = Conv2D(128, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.2)(x)
        
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        outputs = Dense(params.num_classes, activation='sigmoid')(x)  
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(loss=params.loss_obj, optimizer=params.optimizer_obj, metrics=[f2_score])

        
    return model


# In[30]:


def f2_score(y_true, y_pred):
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(tf.round(y_pred), "int32") # implicit 0.5 threshold via tf.round
    y_correct = y_true * y_pred
    sum_true = tf.reduce_sum(y_true, axis=1)
    sum_pred = tf.reduce_sum(y_pred, axis=1)
    sum_correct = tf.reduce_sum(y_correct, axis=1)
    precision = sum_correct / sum_pred
    recall = sum_correct / sum_true
    f_score = 5 * precision * recall / (4 * precision + recall)
    f_score = tf.where(tf.math.is_nan(f_score), tf.zeros_like(f_score), f_score)
    return tf.reduce_mean(f_score)


# In[27]:


def micro_soft_f2(y, y_hat):
    """Modified version of marco_soft_f1 function implemented by Ashref Maiza (ML Engineer @ Amazon). Reference: https://github.com/ashrefm/multi-label-soft-f1/blob/master/Multi-Label%20Image%20Classification%20in%20TensorFlow%202.0.ipynb
    Compute the micro soft F2-score as a cost.
    Average (1 - soft-F2) across all labels.
    Use probability values instead of binary predictions.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=1)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=1)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=1)
    soft_f2 = 5*tp / (5*tp + 4*fn + fp + 1e-16)
    cost = 1 - soft_f2 # reduce 1 - soft-f2 in order to increase soft-f2
    
    micro_cost = tf.reduce_mean(cost) # average on all labels
    
    return micro_cost


# In[28]:


# Modified versions of functions implemented by Ashref Maiza 
def learning_curves(history, version):
    """Plot the learning curves of loss and macro f2 score 
    for the training and validation datasets.
    
    Args:
        history: history callback of fitting a tensorflow keras model 
    """
    path_assets = './reports/assets/{}'.format(version)
    mkdir(path_assets)
    title_loss = 'Training and Validation Loss - Model {}'.format(version)
    title_f2_score = 'Training and Validation Micro F2-score - Model {}'.format(version)
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    micro_f2 = history.history['f2_score']
    val_micro_f2 = history.history['val_f2_score']
    
    epochs = len(loss)

    style.use("bmh")
    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)
    plt.plot(range(1, epochs+1), loss, label='Training Loss')
    plt.plot(range(1, epochs+1), val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.title(title_loss)
    plt.tight_layout()
    
    plt.savefig('./reports/assets/{}/{}.png'.format(version, title_loss))

    plt.subplot(2, 1, 2)
    plt.plot(range(1, epochs+1), micro_f2, label='Training Micro F2-score')
    plt.plot(range(1, epochs+1), val_micro_f2, label='Validation Micro F2-score')
    plt.legend(loc='lower right')
    plt.ylabel('Micro F2-score')
    plt.title(title_f2_score)
    plt.xlabel('epoch')
    plt.tight_layout()
    plt.savefig('./reports/assets/{}/{}.png'.format(version, title_f2_score))
    
    plt.show()
    
    return loss, val_loss, micro_f2, val_micro_f2


def perf_grid(dataset, labels, columns, model, n_thresh=100):
    """Computes the performance table containing target, label names,
    label frequencies, thresholds between 0 and 1, number of tp, fp, fn,
    precision, recall and f-score metrics for each label.
    
    Args:
        dataset (tf.data.Datatset): contains the features array
        labels (numpy array): target matrix of shape (BATCH_SIZE, N_LABELS)
        tags (list of strings): column names in target matrix
        model (tensorflow keras model): model to use for prediction
        n_thresh (int) : number of thresholds to try
        
    Returns:
        grid (Pandas dataframe): performance table 
    """
    
    # Get predictions
    y_hat_val = model.predict(dataset)
    # Define target matrix
    y_val = np.array(labels)
    # Find label frequencies in the validation set
    label_freq = np.array(labels).sum(axis=0)
    # Get label indexes
    label_index = [i for i in range(len(params.tags))]
    # Define thresholds
    thresholds = np.linspace(0,1,n_thresh+1).astype(np.float32)
    
    # Compute all metrics for all labels
    ids, labels, freqs, tps, fps, fns, precisions, recalls, f2s = [], [], [], [], [], [], [], [], []
    for l in label_index:
        for thresh in thresholds:   
            ids.append(l)
            labels.append(columns[l])
            freqs.append(round(label_freq[l]/len(y_val),2))
            y_hat = y_hat_val[:,l]
            y = y_val[:,l]
            y_pred = y_hat > thresh
            tp = np.count_nonzero(y_pred  * y)
            fp = np.count_nonzero(y_pred * (1 - y))
            fn = np.count_nonzero((1 - y_pred) * y)
            precision = tp / (tp + fp + 1e-16)
            recall = tp / (tp + fn + 1e-16)
            f2 = 2*tp / (2*tp + fn + fp + 1e-16)
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
            precisions.append(precision)
            recalls.append(recall)
            f2s.append(f2)
            
    # Create the performance dataframe
    grid = pd.DataFrame({
        'id':ids,
        'label':np.array(labels),
        'freq':freqs,
        'threshold':list(thresholds)*len(label_index),
        'tp':tps,
        'fp':fps,
        'fn':fns,
        'precision':precisions,
        'recall':recalls,
        'f2':f2s})
    
    grid = grid[['id', 'label', 'freq', 'threshold',
                 'tp', 'fn', 'fp', 'precision', 'recall', 'f2']]
    
    return grid


# In[29]:


class ConfusionMatrixCallback(keras.callbacks.Callback):
    def __init__(self, X_test, y_test, params):
        self.X_test = X_test
        self.y_test = y_test
        self.params = params

    def on_epoch_end(self, epoch, logs=None):
        log_folder = './reports/logs'
        log_cm_path = os.path.join(log_folder, 'cm')
        cm_writer = tf.summary.create_file_writer(log_cm_path)
        test_pred = self.model.predict(self.X_test)
        # Calculate the confusion matrix using sklearn.metrics
        cm = tfa.metrics.MultiLabelConfusionMatrix(num_classes=self.params.num_classes)(self.y_test, np.where(test_pred > 0.5, 1, 0))
        figure = plot_confusion_matrix(cm, self.params.columns)
        cm_image = plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        with cm_writer.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)


# In[648]:


def run_baseline_model(version, train_dataset, val_dataset, test_dataset, test_labels, train_params, version_folder):
    if version.startswith('ResNet'):
        model = create_resnet(train_params)
        print('Version: Resnet model - {}'.format(version_folder.split('/')[-1]))

    else:
        model = create_model(version, train_params)
        print('Version: {}'.format(version_folder.split('/')[-1]))
    
    # History
        
    if train_params.rlronplateau:
        print('RLRonPlateau: active\n')
        cm_callback = ConfusionMatrixCallback(test_dataset, test_labels, train_params)
        ReduceLRonPLateau_callback = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, mode='min', min_lr=0.000001)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir= os.path.join(version_folder, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")), histogram_freq=1)

        history = model.fit(train_dataset,
                        epochs=train_params.num_epochs,
                         validation_data=val_dataset,
                        callbacks=[tensorboard_callback,
                                   ReduceLRonPLateau_callback,
                                  cm_callback])
    
    else:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir= os.path.join(version_folder, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")), histogram_freq=1)

        # Confusion matrix
        cm_callback = ConfusionMatrixCallback(test_dataset, test_labels, train_params)
        history = model.fit(train_dataset,
                            epochs=train_params.num_epochs,
                             validation_data=val_dataset,
                            callbacks=[tensorboard_callback,
                                      cm_callback])


    loss, val_loss, micro_f2, val_micro_f2 = learning_curves(history, version)
    grid = perf_grid(test_dataset, test_labels, train_params.columns, model, n_thresh=100)
    
    return model, history, loss, grid


# In[649]:


def run_models(versions, train_dataset, val_dataset, test_dataset, test_labels, train_params, experiment=''):
    v_outputs = {}
    log_folder = './reports/logs'
    log_cm_path = './reports/logs/cm'
    for i, version in enumerate(versions):
        v = []
        v_history = []
        v_loss = []
        v_grid = []
        v_dict = {}
        
        version_folder = os.path.join(log_folder, version+experiment)
        mkdir(log_cm_path)
        v, v_history, v_loss, v_grid = run_baseline_model(version, train_dataset, val_dataset, test_dataset, test_labels, train_params, version_folder)
        shutil.copytree(log_cm_path, os.path.join(version_folder, 'cm'))
        shutil.rmtree(log_cm_path)
        
        v_meta = {'channels': train_params.channels,
                  'image_size': train_params.size,
                 'num_images_train': train_params.num_images_train,
                 'num_images_val': train_params.num_images_val,
                 'num_images_test': train_params.num_images_test,
                 'channels': train_params.channels,
                 'epochs': train_params.num_epochs,
                 'batch_size': train_params.batch_size,
                 'loss_type': train_params.loss_type,
                  'opt_type': train_params.opt_type,
                  'learning_rate': train_params.learning_rate,
                  'momentum': train_params.momentum,
                  'regularization': train_params.regularization,
                  'use_random_flip': train_params.use_random_flip,
                  'use_90rotation': train_params.use_90rotation,
                  'use_elastic_transform': train_params.use_elastic_transform,
                 }
        
        v_dict['meta'] = v_meta
        v_dict['model'] = v
        v_dict['history'] = v_history
        v_dict['loss'] = v_loss
        v_dict['grid'] = v_grid
        
        v_outputs[version] = v_dict
                         
    return v_outputs


# In[1027]:


# When you unfreeze a model that contains BatchNormalization layers in order to do fine-tuning, 
# you should keep the BatchNormalization layers in inference mode by passing training=False 
# when calling the base model. Otherwise the updates applied to the non-trainable weights will suddenly destroy
# what the model has learned.

def create_resnet(params):
    if params.trainable == True:
        print('\n Unfreezing ResNet {}% top layers'.format(params.pcg_unfreeze * 100))
        layers_to_freeze = 175 - int(175*params.pcg_unfreeze) #resnet has 175 layers; this is the number of layers to freeze
        base_model = tf.keras.applications.ResNet50(input_shape=(params.size, params.size, params.channels),
                                                    include_top=False,
                                                    weights='imagenet')
        
        for layer in base_model.layers[:layers_to_freeze]:
            layer.trainable=False
        for layer in base_model.layers[layers_to_freeze:]:
            layer.trainable=True
            
        if params.regularization:
            base_model = add_regularization(base_model, regularizer=tf.keras.regularizers.l2(0.0001))
            print('L2 regularization added')
            
        if params.preprocess:    
            inputs = tf.keras.Input(shape=(params.size, params.size, params.channels))
            x = tf.keras.applications.resnet.preprocess_input(inputs)
            x = base_model(x, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = Dropout(0.2)(x)
            outputs = Dense(params.num_classes, activation='sigmoid')(x)  
            model = tf.keras.Model(inputs, outputs)

            model.compile(loss=params.loss_obj, optimizer=params.optimizer_obj, metrics=[f2_score])
        
        else:
            inputs = tf.keras.Input(shape=(params.size, params.size, params.channels))
            x = base_model(inputs, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = Dropout(0.2)(x)
            outputs = Dense(params.num_classes, activation='sigmoid')(x)  
            model = tf.keras.Model(inputs, outputs)

            model.compile(loss=params.loss_obj, optimizer=params.optimizer_obj, metrics=[f2_score])
            
    elif (params.trainable == 'Full'):
        
        print('\n Using Resnet - Full training'.format(params.pcg_unfreeze))
        base_model = tf.keras.applications.ResNet50(input_shape=(params.size, params.size, params.channels),
                                                    include_top=False,
                                                    weights='imagenet')
                
        if params.preprocess:
            print('\n Using Keras preprocess_input')
            base_model.trainable = True
            if params.regularization:
                base_model = add_regularization(base_model, regularizer=tf.keras.regularizers.l2(0.0001))
                print('L2 regularization added')
            inputs = tf.keras.Input(shape=(params.size, params.size, params.channels))
            x = tf.keras.applications.resnet.preprocess_input(inputs)
            x = base_model(x, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = Dropout(0.2)(x)
            outputs = Dense(params.num_classes, activation='sigmoid')(x)  
            model = tf.keras.Model(inputs, outputs)

            model.compile(loss=params.loss_obj, optimizer=params.optimizer_obj, metrics=[f2_score])
        
        else:
            base_model.trainable = True
            if params.regularization:
                base_model = add_regularization(base_model, regularizer=tf.keras.regularizers.l2(0.0001))
                print('L2 regularization added')
            inputs = tf.keras.Input(shape=(params.size, params.size, params.channels))
            x = tf.keras.applications.resnet.preprocess_input(inputs)
            x = base_model(x, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = Dropout(0.2)(x)
            outputs = Dense(params.num_classes, activation='sigmoid')(x)  
            model = tf.keras.Model(inputs, outputs)

            model.compile(loss=params.loss_obj, optimizer=params.optimizer_obj, metrics=[f2_score])
            
    else:
        print('\n Using Resnet as feature extractor'.format(params.pcg_unfreeze))
        base_model = tf.keras.applications.ResNet50(input_shape=(params.size, params.size, params.channels),
                                                    include_top=False,
                                                    weights='imagenet')
                
        if params.preprocess:
            base_model.trainable = False
            if params.regularization:
                print('L2 regularization added')
                base_model = add_regularization(base_model, regularizer=tf.keras.regularizers.l2(0.0001))
            inputs = tf.keras.Input(shape=(params.size, params.size, params.channels))
            x = tf.keras.applications.mobilenet.preprocess_input(inputs)
            x = base_model(x, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = Dropout(0.2)(x)
            outputs = Dense(params.num_classes, activation='sigmoid')(x)  
            model = tf.keras.Model(inputs, outputs)

            model.compile(loss=params.loss_obj, optimizer=params.optimizer_obj, metrics=[f2_score])
            
        else:
            base_model.trainable = False
            inputs = tf.keras.Input(shape=(params.size, params.size, params.channels))
            x = base_model(inputs, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = Dropout(0.2)(x)
            outputs = Dense(params.num_classes, activation='sigmoid')(x)  
            model = tf.keras.Model(inputs, outputs)

            model.compile(loss=params.loss_obj, optimizer=params.optimizer_obj, metrics=[f2_score])
            
    return model


# In[651]:


# credits to Thalles Silva: https://gist.github.com/sthalles
def add_regularization(model, regularizer=tf.keras.regularizers.l2(0.0001)):

    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
        print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
        return model

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # Save the weights before reloading the model.
    config_json = model.to_json()
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights_resnet.h5')
    model.save_weights(tmp_weights_path)
    
    model = tf.keras.models.model_from_json(config_json)
    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    
    return model


# In[537]:


def run_baseline_model_generator(version, data_generator, test_dataset, test_labels, train_params, version_folder):
    if version.startswith('ResNet'):
        model = create_resnet(train_params)
        print('Version: Resnet model - {}'.format(version_folder.split('/')[-1]))
        if train_params.regularization:
            model = add_regularization(model, regularizer=tf.keras.regularizers.l2(0.0001))
            print('L2 regularization added')
    else:
        model = create_model(version, train_params)
        print('Version: {}'.format(version_folder.split('/')[-1]))
    
    # History
        
    if train_params.rlronplateau:
        print('RLRonPlateau: active\n')
        cm_callback = ConfusionMatrixCallback(test_dataset, test_labels, train_params)
        ReduceLRonPLateau_callback = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, mode='min', min_lr=0.000001)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir= os.path.join(version_folder, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")), histogram_freq=1)

        history = model.fit_generator(data_generator['train'],
                            steps_per_epoch=train_params.num_images_train// train_params.batch_size,
                            epochs=train_params.num_epochs,
                            validation_data=data_generator['val'], 
                            validation_steps=train_params.num_images_val// train_params.batch_size,
                            callbacks=[tensorboard_callback, cm_callback, ReduceLRonPLateau_callback])
    
    else:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir= os.path.join(version_folder, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")), histogram_freq=1)

        # Confusion matrix
        cm_callback = ConfusionMatrixCallback(test_dataset, test_labels, train_params)
        history = model.fit_generator(data_generator['train'],
                            steps_per_epoch=train_params.num_images_train// train_params.batch_size,
                            epochs=train_params.num_epochs,
                            validation_data=data_generator['val'], 
                            validation_steps=train_params.num_images_val// train_params.batch_size,
                            callbacks=[tensorboard_callback, cm_callback])


    loss, val_loss, micro_f2, val_micro_f2 = learning_curves(history, version)
    grid = perf_grid(test_dataset, test_labels, train_params.columns, model, n_thresh=100)
    
    return model, history, loss, grid


# In[541]:


def run_models_generator(versions, data_generator, test_dataset, test_labels, train_params, experiment=''):
    v_outputs = {}
    log_folder = './reports/logs'
    log_cm_path = './reports/logs/cm'
    for i, version in enumerate(versions):
        v = []
        v_history = []
        v_loss = []
        v_grid = []
        v_dict = {}
        
        version_folder = os.path.join(log_folder, version+experiment)
        mkdir(log_cm_path)
        v, v_history, v_loss, v_grid = run_baseline_model_generator(version, data_generator, test_dataset, test_labels, train_params, version_folder)
        shutil.copytree(log_cm_path, os.path.join(version_folder, 'cm'))
        shutil.rmtree(log_cm_path)
        
        v_meta = {'channels': train_params.channels,
                  'image_size': train_params.size,
                 'num_images_train': train_params.num_images_train,
                 'num_images_val': train_params.num_images_val,
                 'num_images_test': train_params.num_images_test,
                 'channels': train_params.channels,
                 'epochs': train_params.num_epochs,
                 'batch_size': train_params.batch_size,
                 'loss_type': train_params.loss_type,
                  'opt_type': train_params.opt_type,
                  'learning_rate': train_params.learning_rate,
                  'momentum': train_params.momentum,
                'regularization': train_params.regularization,
                  'use_random_flip': train_params.use_random_flip,
                  'use_90rotation': train_params.use_90rotation,
                  'use_elastic_transform': train_params.use_elastic_transform,
                 }
        
        v_dict['meta'] = v_meta
        v_dict['model'] = v
        v_dict['history'] = v_history
        v_dict['loss'] = v_loss
        v_dict['grid'] = v_grid
        
        v_outputs[version] = v_dict
                         
    return v_outputs


# In[822]:


def plot_f2_label_threshold(df, label, versions, experiment):
    assets_path = './reports/assets/'
    save_path = os.path.join(assets_path, list(versions.keys())[0] + experiment)

    f2_max = df[df['label'] == label]['f2'].max(axis=0)
    ix_max = df[df['label'] == label]['f2'].idxmax()
    thr_max = df.loc[ix_max]['threshold']

    # Plot
    font = {'size': 18}
    plt.rc('font', **font)
    fig, ax = plt.subplots(1, 1, dpi=200)
    
    ax.plot(thr_max, f2_max, 'ko', markersize=20, fillstyle='none', markeredgewidth=1.5, color='r', alpha=0.8)
    ax.annotate('f2_max: {:.2f}'.format(f2_max), (thr_max, f2_max), color='r', alpha=.8)
    ax.set_xticks(np.linspace(0,1), 10)
    ax.set_yticks(np.linspace(0, round(f2_max + 0.1 * f2_max, 1), 10))
    ax.set_ylim(top=f2_max + 0.1*f2_max)
    sns.lineplot(x='threshold', y='f2', data=df[df['label'] == label], ax=ax, color='g', alpha=.8)
    plt.title('f2 - threshold curve for {}'.format(label), size=18)
    ax.set_facecolor('w')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, label))


# In[823]:


def plot_f2_labels(versions, experiment):
    df = pd.DataFrame(versions['ResNet']['grid'])
    labels = df['label'].unique()
    for label in labels:
        plot_f2_label_threshold(df, label, versions, experiment)


# In[834]:


def results_to_file(versions, experiment):
    # save
    assets_path = './reports/assets/'
    saved_models_dir = './reports/saved_models'

    save_path = os.path.join(assets_path, list(versions.keys())[0] + experiment)
    save_meta_csv_path = os.path.join(save_path, list(versions.keys())[0] + experiment + '_meta_.csv')
    save_grid_csv_path = os.path.join(save_path, list(versions.keys())[0] + experiment + '_grid_.csv')
    mkdir(save_path)

    df_meta = pd.DataFrame(versions['ResNet']['meta'], index=[0])
    df_grid = pd.DataFrame(versions['ResNet']['grid'])

    # save meta and grid to csv
    pd.DataFrame.to_csv(df_meta, save_meta_csv_path, index=False)
    pd.DataFrame.to_csv(df_grid, save_grid_csv_path, index=False)

    # save model
    versions['ResNet']['model'].save(os.path.join(saved_models_dir, list(versions.keys())[0] + experiment))


# In[1041]:


def df_max_f2_label(version_output):
    df = version_output['ResNet']['grid']
    idx = []
    for label in df['label'].unique():
        idx.append(df[df['label'] == label]['f2'].idxmax())
    return df.loc[idx]

