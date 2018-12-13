#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:43:52 2017

@author: leminen
"""
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import shlex
from tensorflow.contrib import slim
from tensorflow.contrib.layers.python.layers import layers as layers_lib

import src.utils as utils
import src.data.util_data as util_data
import src.data.datasets.psd as psd_dataset

layers = tf.contrib.layers
framework = tf.contrib.framework
ds = tf.contrib.distributions


def hparams_parser_train(hparams_string):
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch_max', 
                        type=int, default='100', 
                        help='Max number of epochs to run')

    parser.add_argument('--batch_size', 
                        type=int, default='64', 
                        help='Number of samples in each batch')

    parser.add_argument('--use_imagenet',
                        type=bool, default=True,
                        help='Use pretrained model trained for imagenet for weight initilization')

    parser.add_argument('--model_version',
                        type=str, default='VGG16',
                        choices=['VGG16',
                                 'VGG19'],
                        help='Choose VGG model configuration')

    ## add more model parameters to enable configuration from terminal
    
    return parser.parse_args(shlex.split(hparams_string))


def hparams_parser_evaluate(hparams_string):
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch_no', 
                        type=int,
                        default=None, 
                        help='Epoch no to reload')

    ## add more model parameters to enable configuration from terminal

    return parser.parse_args(shlex.split(hparams_string))


class VGG(object):
    def __init__(self, dataset, id):

        self.model = 'VGG'
        if id != None:
            self.model = self.model + '_' + id

        self.dir_base        = 'models/' + self.model
        self.dir_logs        = self.dir_base + '/logs'
        self.dir_checkpoints = self.dir_base + '/checkpoints'
        self.dir_results     = self.dir_base + '/results'
        
        utils.checkfolder(self.dir_checkpoints)
        utils.checkfolder(self.dir_logs)
        utils.checkfolder(self.dir_results)

        self.dataset = dataset
        # Specify valid dataset for model
        if dataset =='PSD_Segmented':
            self.dateset_filenames = ['data/processed/PSD_Segmented/PSD-data_{:03d}-of-{:03d}.tfrecord'.format(i+1,psd_dataset._NUM_SHARDS) for i in range(psd_dataset._NUM_SHARDS)]
            self.lbls_dim = 9
            self.image_dims = [128,128,3]
            self.fc6_dims = [4,4] # 128/(2^5) = 4

        else:
            raise ValueError('Selected Dataset is not supported by model: ' + self.model)
        
       
    def _create_inference(self, inputs, is_training = True, dropout_keep_prob = 0.5):
        """ Define the inference model for the network
        Args:
    
        Returns:
        """
        if self.model_version == 'VGG16':
            with tf.variable_scope('vgg_16'):
                with framework.arg_scope([layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d]):
                    # From tensorflow.contrib.slim.nets.vgg.vgg_16
                    net = layers_lib.repeat(inputs, 2, layers.conv2d, 64, [3, 3], scope='conv1')
                    net = layers_lib.max_pool2d(net, [2, 2], scope='pool1')
                    net = layers_lib.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv2')
                    net = layers_lib.max_pool2d(net, [2, 2], scope='pool2')
                    net = layers_lib.repeat(net, 3, layers.conv2d, 256, [3, 3], scope='conv3')
                    net = layers_lib.max_pool2d(net, [2, 2], scope='pool3')
                    net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv4')
                    net = layers_lib.max_pool2d(net, [2, 2], scope='pool4')
                    net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv5')
                    net = layers_lib.max_pool2d(net, [2, 2], scope='pool5')

                    # Custom fc layers to allow variable input image size
                    net = layers.conv2d(net, 4096, self.fc6_dims, padding='VALID', scope='fc6')
                    net = layers_lib.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
                    net = layers.conv2d(net, 4096, [1, 1], scope='fc7')
                    net = layers_lib.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')
                    net = layers.conv2d(net, self.lbls_dim, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')

                    logits = net

        elif self.model_version == 'VGG19':
            with tf.variable_scope('vgg_19'):
                with framework.arg_scope([layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d]):
                    # From tensorflow.contrib.slim.nets.vgg.vgg_19
                    net = layers_lib.repeat(inputs, 2, layers.conv2d, 64, [3, 3], scope='conv1')
                    net = layers_lib.max_pool2d(net, [2, 2], scope='pool1')
                    net = layers_lib.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv2')
                    net = layers_lib.max_pool2d(net, [2, 2], scope='pool2')
                    net = layers_lib.repeat(net, 4, layers.conv2d, 256, [3, 3], scope='conv3')
                    net = layers_lib.max_pool2d(net, [2, 2], scope='pool3')
                    net = layers_lib.repeat(net, 4, layers.conv2d, 512, [3, 3], scope='conv4')
                    net = layers_lib.max_pool2d(net, [2, 2], scope='pool4')
                    net = layers_lib.repeat(net, 4, layers.conv2d, 512, [3, 3], scope='conv5')
                    net = layers_lib.max_pool2d(net, [2, 2], scope='pool5')

                    # Custom fc layers to allow variable input image size
                    net = layers.conv2d(net, 4096, self.fc6_dims, padding='VALID', scope='fc6')
                    net = layers_lib.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
                    net = layers.conv2d(net, 4096, [1, 1], scope='fc7')
                    net = layers_lib.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')
                    net = layers.conv2d(net, self.lbls_dim, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')

                    logits = net

        return logits
    
    def _create_losses(self, logits, labels):
        """ Define loss function[s] for the network
        Args:
    
        Returns:
        """
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=labels,
                logits=logits,
                name = 'Loss')
        )
        return loss
        
    def _create_optimizer(self, loss):
        """ Create optimizer for the network
        Args:
    
        Returns:
        """
        model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        optimizer = tf.train.AdamOptimizer()
        optimizer_op = optimizer.minimize(loss, var_list = model_vars)

        return optimizer_op
        
    def _create_summaries(self, loss):
        """ Create summaries for the network
        Args:
    
        Returns:
        """
        
        ### Add summaries
        with tf.name_scope("summaries"):
            tf.summary.scalar('model_loss', loss) # placeholder summary
            summary_op = tf.summary.merge_all()

        return summary_op
        
        
    def train(self, hparams_string):
        """ Run training of the network
        Args:
    
        Returns:
        """

        args_train = hparams_parser_train(hparams_string)
        self.batch_size = args_train.batch_size
        self.epoch_max = args_train.epoch_max 
        self.use_imagenet = args_train.use_imagenet
        self.model_version = args_train.model_version

        utils.save_model_configuration(args_train, self.dir_base)
        
        # Use dataset for loading in datasamples from .tfrecord (https://www.tensorflow.org/programmers_guide/datasets#consuming_tfrecord_data)
        # The iterator will get a new batch from the dataset each time a sess.run() is executed on the graph.
        dataset = tf.data.TFRecordDataset(self.dateset_filenames)
        dataset = dataset.map(util_data.decode_image)      # decoding the tfrecord
        dataset = dataset.map(self._preProcessData)        # potential local preprocessing of data
        dataset = dataset.shuffle(buffer_size = 10000, seed = None)
        dataset = dataset.batch(batch_size = self.batch_size)
        iterator = dataset.make_initializable_iterator()
        input_getBatch = iterator.get_next()

        input_images = tf.placeholder(
            dtype = tf.float32, 
            shape = [None] + self.image_dims, 
            name = 'input_images')
        input_lbls = tf.placeholder(
            dtype = tf.float32,   
            shape = [None, self.lbls_dim], 
            name = 'input_lbls')

        
        # define model, loss, optimizer and summaries.
        output_logits = self._create_inference(input_images)
        loss = self._create_losses(output_logits, input_lbls)
        optimizer_op = self._create_optimizer(loss)
        summary_op = self._create_summaries(loss)
        
        # show network architecture
        utils.show_all_variables()

        if self.use_imagenet:
            if self.model_version == 'VGG16':
                path_imagenet_ckpt = os.path.join(self.dir_checkpoints, 'vgg_16.ckpt')
                if not tf.gfile.Exists(path_imagenet_ckpt):
                    url_imagenet_model = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"
                    utils.download_and_uncompress_tarball(url_imagenet_model,self.dir_checkpoints)

                variables_to_restore = slim.get_model_variables('vgg_16')
                variables_to_restore = variables_to_restore[:-6] # ignore fc layers
                init_fn = slim.assign_from_checkpoint_fn(path_imagenet_ckpt, variables_to_restore)

            elif self.model_version == 'VGG19':
                path_imagenet_ckpt = os.path.join(self.dir_checkpoints, 'vgg_19.ckpt')
                if not tf.gfile.Exists(path_imagenet_ckpt):
                    url_imagenet_model = "http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz"
                    utils.download_and_uncompress_tarball(url_imagenet_model,self.dir_checkpoints)

                variables_to_restore = slim.get_model_variables('vgg_19')
                variables_to_restore = variables_to_restore[:-6] # ignore fc layers
                init_fn = slim.assign_from_checkpoint_fn(path_imagenet_ckpt, variables_to_restore)
        
        with tf.Session() as sess:
            
            # Initialize all model Variables.
            sess.run(tf.global_variables_initializer())

            if self.use_imagenet:
                init_fn(sess)
                
            
            # Create Saver object for loading and storing checkpoints
            saver = tf.train.Saver()
            
            # Create Writer object for storing graph and summaries for TensorBoard
            writer = tf.summary.FileWriter(self.dir_logs, sess.graph)
            
            
            # Reload Tensor values from latest checkpoint
            ckpt = tf.train.get_checkpoint_state(self.dir_checkpoints)
            epoch_start = 0
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                epoch_start = int(ckpt_name.split('-')[-1])
            
            interationCnt = 0
            # Do training loops
            for epoch_n in range(epoch_start, self.epoch_max):

                # Initiate or Re-initiate iterator
                sess.run(iterator.initializer)

                # Test model output before any training
                # if epoch_n == 0:
                #     summary_loss = sess.run(summary_op)
                #     writer.add_summary(summary_loss, global_step=-1)
                
                utils.show_message('Running training epoch no: {0}'.format(epoch_n))
                while True:
                    try:
                        image_batch, lbl_batch = sess.run(input_getBatch)
                        _, summary_loss = sess.run(
                            [optimizer_op, summary_op],
                            feed_dict={input_images:    image_batch,
                                       input_lbls:      lbl_batch})
                        
                        writer.add_summary(summary_loss, global_step=interationCnt)
                        counter =+ 1
                        
                    except tf.errors.OutOfRangeError:
                        # Do some evaluation after each Epoch
                        break
                
                if epoch_n % 1 == 0:
                    saver.save(sess,os.path.join(self.dir_checkpoints, self.model + '.model'), global_step=epoch_n)
                
            
    
    def evaluate(self, hparams_string):
        """ Run prediction of the network
        Args:
    
        Returns:
        """
        
        args_evaluate = hparams_parser_evaluate(hparams_string)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
    

    def _preProcessData(self, image_proto, lbl_proto, class_proto, height_proto, width_proto, channels_proto, origin_proto):
        """ Local preprocessing of data from dataset
        also used to select which elements to parse onto the model
        Args:
          all outputs of util_data.decode_image

        Returns:
        """

        image = image_proto

        # Dataset specific preprocessing
        if self.dataset == 'MNIST':
            pass

        elif self.dataset == 'PSD_Nonsegmented':
            pass

        elif self.dataset == 'PSD_Segmented':
            max_dim = psd_dataset._LARGE_IMAGE_DIM
            height_diff = max_dim - height_proto
            width_diff = max_dim - width_proto

            paddings = tf.floor_div([[height_diff, height_diff], [width_diff, width_diff], [0,0]],2)
            image = tf.pad(image, paddings, mode='CONSTANT', name=None, constant_values=-1)

        image = tf.image.resize_images(image, size = self.image_dims[0:-1])  

        lbl = tf.one_hot(lbl_proto, self.lbls_dim)

        return image, lbl
