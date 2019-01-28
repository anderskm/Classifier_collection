#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:43:52 2017

@author: leminen
"""
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import shlex
from tensorflow.contrib import slim
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.python import pywrap_tensorflow
import math
import numpy as np

# For preprocessing_factory
# import ast
# import inspect

import src.confusionmatrix as CM

import src.utils as utils
import src.preprocess_factory as preprocess_factory
import src.tf_custom_summaries as tf_custom_summaries
import src.data.util_data as util_data
# import src.data.datasets.psd as psd_dataset
# import src.data.datasets.seeds as seeds_dataset

import src.data.datasets.DS_PSDs_no_grass as DS_PSDs
# import src.data.datasets.DS_Seeds_abnormal as DS_Seeds
# import src.data.datasets.DS_Seeds_D0 as DS_Seeds_D0
# import src.data.datasets.DS_Barley_Next as DS_Barley_Next
# import src.data.datasets.DS_Barley_Next_Stratified as DS_Barley_Next_Stratified

import src.models.resnet_utils as resnet_utils
import src.models.resnet_v1 as resnet_v1
# import src.models.resnet_v2 as resnet_v2

layers = tf.contrib.layers
framework = tf.contrib.framework
ds = tf.contrib.distributions

def str2bool(v):
    if v.lower() in ('yes', 'true','y','t','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def hparams_parser_train(hparams_string):
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch_max', 
                        type=int, default='100', 
                        help='Max number of epochs to run')

    parser.add_argument('--batch_size', 
                        type=int, default='10',
                        help='Number of samples in each batch')

    parser.add_argument('--model_version',
                        type=str, default='ResNet50',
                        choices=['ResNet50',
                                 'ResNet101',
                                 'ResNet152',
                                 'ResNet200'],
                        help='Choose VGG model configuration')

    parser_group_pretrain = parser.add_argument_group('Pretraining options')
    parser_group_pretrain.add_argument('--pretrained_model',
                                        type=str, default='',
                                        help='Specify the path to a pretrained model. When specified, all variables matching variables in the netword are loaded into the network.')
    parser_group_pretrain.add_argument('--pretrain_exclude_input',
                                        type=str2bool, default='False',
                                        help='Exclude the input layer, when loading pretrained model.')
    parser_group_pretrain.add_argument('--pretrain_exclude_output',
                                        type=str2bool, default='True',
                                        help='Exclude the output layer, when loading pretrained model.')

    parser_group_pretrain.add_argument('--optim_vars',
                                        type=str, default='all',
                                        choices=['all',
                                                'non_restored'],
                                        help='Specify, which variables to optimize.')

                        # shuffle_before_split=True, shuffle_seed=1337, group_before_split=False, validation_method='none', holdout_split=[0.8, 0.1, 0.1], cross_folds=10, cross_val_folds=[], cross_test_folds=[0], stratify_training_set = True):
    parser_group_dataset = parser.add_argument_group('Dataset options')
    parser_group_dataset.add_argument('--shuffle_before_split',
                                        type=str2bool, default='True',
                                        help='Shuffle all examples before splitting dataset into training, validation and test.')
    parser_group_dataset.add_argument('--shuffle_seed',
                                        type=int,default=1337,
                                        help='Set the random seed using in the random shuffle (if shuffle_before_split is True).')
    parser_group_dataset.add_argument('--group_before_split',
                                        type=str2bool, default='True',
                                        help='Group examples according to grouping function before splitting dataset into training, validation and test.')
    parser_group_dataset.add_argument('--stratify_training_set',
                                        type=str2bool, default='False',
                                        help='Stratify the training set by copying examples to make class distributions approximately equal (within a factor of 0.5 to 1.5 of the most represented class).')
    parser_group_dataset.add_argument('--validation_method',
                                        type=str, default='none',
                                        choices=['none',
                                                'holdout',
                                                'shards'],
                                        help='Select validation method used for splitting dataset into training, validation and test. Choices: {%(choices)s}. \'none\': use all examples for training. \'holdout\': split dataset into training, validation and test')
    parser_group_dataset.add_argument('--holdout_split',
                                        type=float, nargs=3, default=[0.8, 0.1, 0.1],
                                        help='Relative fractions used when splitting dataset using validation_method == holdout. Must be 3 numbers ({%(type)s}) separated by spaces. E.g. --holdout_split 0.8 0.1 0.1 . Default: %(default)s')
    parser_group_dataset.add_argument('--shard_val',
                                        type=int, default=None,
                                        help='Index of shard to use for validation set, when validation_method == shards. Default: %(default)s')
    parser_group_dataset.add_argument('--shard_test',
                                        type=int, default=0,
                                        help='Index of shard to use for test set, when validation_method == shards. Default: %(default)s')
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


class ResNet(object):
    def __init__(self, dataset, id):

        self.model = 'ResNet'
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
        print('Selected dataset: ' + dataset)
        # Specify valid dataset for model
        if dataset =='PSD_Segmented':
            # self.dateset_filenames = ['data/processed/PSD_Segmented/PSD-data_{:03d}-of-{:03d}.tfrecord'.format(i+1,psd_dataset._NUM_SHARDS) for i in range(psd_dataset._NUM_SHARDS)]
            self.lbls_dim = 12
            self.image_dims = [128,128,3]
            self.fc_dims = [4,4] # 128/(2^5) = 4
        elif dataset == 'seeds':
            # self.dateset_filenames = ['data/processed/seeds_all/PSD-data_{:03d}-of-{:03d}.tfrecord'.format(i+1,psd_dataset._NUM_SHARDS) for i in range(psd_dataset._NUM_SHARDS)]
            self.lbls_dim = 2
            self.image_dims = [224,224,19]
            self.fc_dims = [7,7] # 128/(2^5) = 4
        elif dataset == 'seeds_all':
            # self.dateset_filenames = ['data/processed/seeds_all/PSD-data_{:03d}-of-{:03d}.tfrecord'.format(i+1,psd_dataset._NUM_SHARDS) for i in range(psd_dataset._NUM_SHARDS)]
            self.lbls_dim = 2
            self.image_dims = [256,256,19]
            self.fc_dims = [8,8] # 128/(2^5) = 4
        elif dataset == 'barley_d0':
            # self.dateset_filenames = ['data/processed/seeds_all/PSD-data_{:03d}-of-{:03d}.tfrecord'.format(i+1,psd_dataset._NUM_SHARDS) for i in range(psd_dataset._NUM_SHARDS)]
            self.lbls_dim = 2
            self.image_dims = [256,256,19]
            self.fc_dims = [8,8] # 128/(2^5) = 4
        elif dataset == 'barley_next':
            # self.dateset_filenames = ['data/processed/seeds_all/PSD-data_{:03d}-of-{:03d}.tfrecord'.format(i+1,psd_dataset._NUM_SHARDS) for i in range(psd_dataset._NUM_SHARDS)]
            self.lbls_dim = 2
            self.image_dims = [256,256,19]
            self.fc_dims = [8,8] # 128/(2^5) = 4
        elif dataset == 'barley_next_stratified':
            self.lbls_dim = 2
            self.image_dims = [256,256,19]
            self.fc_dims = [8,8] # 128/(2^5) = 4
        else:
            raise ValueError('Selected Dataset is not supported by model: ' + self.model)

    def _get_variable_names_from_checkpoint(self, file_name):
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        variable_names = []
        for key in sorted(var_to_shape_map):
            variable_names.append(key)
        return variable_names

    def _load_pretrained_model(self, output_logits, pretrained_model_path, variables_to_exclude=[]):
        utils.show_message('Loading pretrained model:', lvl=1)

        print('Pre-trained model path       :' + pretrained_model_path)

        # # Get all restorable variables from graph
        graph_variables_all = slim.get_variables_to_restore()
        graph_variable_names_all = [tf_var.name.split(':')[0] for tf_var in graph_variables_all]
        print('Variables in graph           : ' + '{:>7d}'.format(len(graph_variable_names_all)))

        # Get variable names from pretrained model
        pretrained_model_variables_names_all = self._get_variable_names_from_checkpoint(pretrained_model_path)
        print('Variables in pretrained model: ' + '{:>7d}'.format(len(pretrained_model_variables_names_all)))

        # Get intersection of variable names in graph and pretrained model
        # variable_names_in_graph_and_pretrained_model = list(set(graph_variable_names_all).intersection(set(pretrained_model_variables_names_all)))
        # print('Variables intersecting       : ' + '{:>7d}'.format(len(variable_names_in_graph_and_pretrained_model)))

        # Add variable names of "not in pretrained model, but in graph" to exclude list
        variables_to_exclude += list(set(graph_variable_names_all)-set(pretrained_model_variables_names_all))

        # # Find and exclude biases (since they do not appear to be saved in the  imagenet checkpoint)
        # self.tf_variables_to_exclude = slim.get_variables_by_suffix("biases")
        # self.variables_to_exclude = [tf_var.name for tf_var in self.tf_variables_to_exclude]

        # # Find and exclude input and output layers
        # self.variables_to_exclude += ['resnet_v1_50/conv1','resnet_v1_50/logits']
        # self.variables_to_exclude += ['resnet_v1_50/logits']
        # print('Variables to exclude         : ' + '{:>7d}'.format(len(variables_to_exclude)))

        model_vars_restored = slim.get_variables_to_restore(exclude=variables_to_exclude)
        model_var_names_restored = [tf_var.name.split(':')[0] for tf_var in model_vars_restored]
        print('Variables to restore         : ' + '{:>7d}'.format(len(model_var_names_restored)))
        model_vars_not_restored = list(set(graph_variables_all) - set(model_vars_restored))
        print('Variables not restored       : ' + '{:>7d}'.format(len(model_vars_not_restored)))

        # self.variables_to_restore = slim.get_variables_to_restore()
        restorer = tf.train.Saver(model_vars_restored)
        # restorer = tf.train.Saver()
        with tf.Session() as sess:
            # restorer.restore(sess, "C:/github/Classifier_collection/models/resnet_v1_50.ckpt")
            # print('Latest check-point:')
            # print(tf.train.latest_checkpoint("C:/github/Classifier_collection/models/ResNet_batch_train_test/checkpoints"))
            # print('That was latest check-point!')
            # restorer.restore(sess, "C:/github/Classifier_collection/models/ResNet_batch_train_test/checkpoints/ResNet_batch_train_test.model-55")
            restorer.restore(sess, pretrained_model_path)

        return output_logits, model_vars_restored, model_vars_not_restored
        
       
    def _create_inference(self, inputs, num_classes, is_training = True, dropout_keep_prob = 0.5):
        """ Define the inference model for the network
        Args:
    
        Returns:
        """
        utils.show_message('Create model inference', lvl=1)
        print('Model: ' + self.model_version)

        if self.model_version == 'ResNet50':
            logits, endpoints = resnet_v1.resnet_v1_50(inputs, num_classes, is_training=is_training, global_pool=True, spatial_squeeze=False)
            input_layer_name = ['resnet_v1_50/conv1']
            output_layer_names = [ep for ep in endpoints if ('logits' in ep)]

        elif self.model_version == 'ResNet101':
            logits, endpoints = resnet_v1.resnet_v1_101(inputs, num_classes, is_training=is_training, spatial_squeeze=False)
            input_layer_name = ['resnet_v1_101/conv1']
            output_layer_names = [ep for ep in endpoints if ('logits' in ep)]
        
        elif self.model_version == 'ResNet152':
            logits, endpoints = resnet_v1.resnet_v1_152(inputs, num_classes, is_training=is_training, spatial_squeeze=False)
            input_layer_name = ['resnet_v1_152/conv1']
            output_layer_names = [ep for ep in endpoints if ('logits' in ep)]

        elif self.model_version == 'ResNet200':
            logits, endpoints = resnet_v1.resnet_v1_200(inputs, num_classes, is_training=is_training, spatial_squeeze=False)
            input_layer_name = ['resnet_v1_200/conv1']
            output_layer_names = [ep for ep in endpoints if ('logits' in ep)]

        print('Input layer  : ' + '; '.join(input_layer_name))
        print('Output layers: ' + '; '.join(output_layer_names))

        return logits, endpoints, input_layer_name, output_layer_names
    
    def _create_losses(self, logits, labels, num_classes):
        """ Define loss function[s] for the network
        Args:
    
        Returns:
        """
        loss = tf.constant(0, dtype=tf.float32)
        for logit, label, N_classes in zip(logits,labels, num_classes):
            lbl = tf.one_hot(label, N_classes)
            loss += tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=lbl,
                        logits=logit,
                        name = 'Loss')
        loss = tf.reduce_mean(loss)

        # loss = tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits_v2(
        #         labels=labels,
        #         logits=logits,
        #         name = 'Loss')
        # )

        # targets = tf.cast(tf.math.argmax(labels,axis=1), tf.float32)
        # _logits = tf.cast(tf.squeeze(tf.math.reduce_max(logits, axis=3)), tf.float32)
        # loss = tf.reduce_mean(
        #     tf.nn.weighted_cross_entropy_with_logits(
        #         targets=targets,
        #         logits=_logits,
        #         pos_weight=0.1)
        # )
        return loss
        
    def _create_optimizer(self, loss, variables_to_optimize=None):
        """ Create optimizer for the network
        Args:
    
        Returns:
        """

        utils.show_message('Setup optimizer', lvl=1)

        model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        if (variables_to_optimize is not None):
            model_vars_train = variables_to_optimize #[tensor for tensor in model_vars if (tensor.name in variable_names_to_optimize)]
        else:
            model_vars_train = model_vars
        print('Number of tensors to optimize: ' + str(len(model_vars_train)))
        print('Tensors to optimize: ')
        print([T.name for T in model_vars_train])

        optimizer = tf.train.AdamOptimizer()
        optimizer_op = optimizer.minimize(loss, var_list = model_vars_train)

        return optimizer_op
        
    def _create_summaries(self, loss, summary_dict={}, summary_list=[]):
        """ Create summaries for the network
        Args:
    
        Returns:
        """

        # TODO: Custom histogram with per-class performance
        # See: https://stackoverflow.com/questions/42012906/create-a-custom-tensorflow-histogram-summary
        
        ### Add summaries
        # with tf.name_scope("summaries"):
            # tf.summary.scalar('model_loss', loss) # placeholder summary
            
        for name, tf_placeholder in summary_dict.items():
            # Inspired by:
            # https://stackoverflow.com/a/41031284
            tf.summary.scalar(name, tf_placeholder)

        for tf_placeholder in summary_list:
            tf.summary.scalar(tf_placeholder.name, tf_placeholder)

        return

    def _show_progress(self, tag, epoch, batch_counter, batch_max, loss, CMats):
        # print('T' + '{:d}'.format(epoch_n) + ' ' + '{:>4d}'.format(batchCounter)  + ' Loss: ' + '{:>7.3g}'.format(loss_out) + ' Acc(s): ' + '  '.join(['{:>5.3f}'.format(CMat.accuracy()) for CMat in CMatsTrain]))
        output_string = tag + '{:d}'.format(epoch) + ' ' + '{:>4d}'.format(batch_counter) + '/' + '{:>4d}'.format(batch_max)  + ' Loss: ' + '{:>7.3g}'.format(loss) + ' Acc(s): ' + '  '.join(['{:>5.3f}'.format(CMat.accuracy()) for CMat in CMats])
        sys.stdout.write('\routput_string)
        sys.stdout.flush()
        
    def train(self, hparams_string, preprocessing_params=''):
        """ Run training of the network
        Args:
    
        Returns:
        """

        args_train = hparams_parser_train(hparams_string)
        self.batch_size = args_train.batch_size
        self.epoch_max = args_train.epoch_max 
        self.model_version = args_train.model_version
        pretrained_model_path = args_train.pretrained_model
        use_pretrained_model = False if pretrained_model_path is '' else True
        pretrain_exclude_input = args_train.pretrain_exclude_input
        pretrain_exclude_output = args_train.pretrain_exclude_output
        optim_vars = args_train.optim_vars
        args_train.preprocessing = preprocessing_params


        print('Training parameters:')
        print(args_train)

        utils.save_model_configuration(args_train, self.dir_base)
        
        # Load dataset
        if (self.dataset == 'PSD_Segmented'):
            DS = DS_PSDs.Dataset()
        elif (self.dataset == 'seeds_all'):
            DS = DS_Seeds.Dataset()
        elif (self.dataset == 'barley_d0'):
            DS = DS_Seeds_D0.Dataset()
        elif (self.dataset == 'barley_next'):
            DS = DS_Barley_Next.Dataset()
        elif (self.dataset == 'barley_next_stratified'):
            DS = DS_Barley_Next_Stratified.Dataset()
        tf_dataset_list, dataset_sizes = DS.get_dataset_list(shuffle_before_split=args_train.shuffle_before_split,
                                                            shuffle_seed=args_train.shuffle_seed,
                                                            group_before_split=args_train.group_before_split,
                                                            validation_method=args_train.validation_method,
                                                            holdout_split=args_train.holdout_split,
                                                            cross_folds=10,
                                                            cross_val_fold=None,
                                                            cross_test_fold=0,
                                                            shard_val=args_train.shard_val,
                                                            shard_test=args_train.shard_test,
                                                            stratify_training_set=args_train.stratify_training_set)

        class_dicts = DS.get_class_dicts()
        num_classes = [len(class_dict) for class_dict in class_dicts]

        preprocessing = preprocess_factory.preprocess_factory()
        if not (preprocessing_params == ''):
            # Setup preprocessing pipeline
            preprocessing.prep_pipe_from_string(preprocessing_params)

        with tf.name_scope('Training_dataset'):
            tf_dataset_train = tf_dataset_list[0]
            tf_dataset_train = tf_dataset_train.shuffle(buffer_size = 10000, seed = None)
            tf_dataset_train = tf_dataset_train.map(DS._decode_from_TFexample)
            # tf_dataset_train = tf_dataset_train.map(self._preProcessData)        # potential local preprocessing of data
            tf_dataset_train = tf_dataset_train.map(preprocessing.pipe)
            tf_dataset_train = tf_dataset_train.batch(batch_size = self.batch_size, drop_remainder=False)
            tf_dataset_train = tf_dataset_train.repeat(count=-1) # -1 --> repeat indefinitely
            tf_dataset_train = tf_dataset_train.prefetch(buffer_size=3)
            tf_dataset_train_iterator = tf_dataset_train.make_one_shot_iterator()
            # tf_dataset_train_iterator = tf_dataset_train.make_initializable_iterator()
            input_getBatch = tf_dataset_train_iterator.get_next()

        with tf.name_scope('Validation_dataset'):
            tf_dataset_val = tf_dataset_list[1]
            if (tf_dataset_val is not None):
                tf_dataset_val = tf_dataset_val.map(DS._decode_from_TFexample)
                # tf_dataset_val = tf_dataset_val.map(self._preProcessData)        # potential local preprocessing of data
                tf_dataset_val = tf_dataset_val.map(preprocessing.pipe)
                tf_dataset_val = tf_dataset_val.batch(batch_size = self.batch_size, drop_remainder=False)
                tf_dataset_val = tf_dataset_val.repeat(count=-1) # -1 --> repeat indefinitely
                tf_dataset_val = tf_dataset_val.prefetch(buffer_size=3)
                # tf_dataset_val_iterator = tf_dataset_val.make_initializable_iterator()
                tf_dataset_val_iterator = tf_dataset_val.make_one_shot_iterator()
                tf_input_getBatch_val = tf_dataset_val_iterator.get_next()

        input_images = tf.placeholder(
            dtype = tf.float32, 
            shape = [None] + self.image_dims, 
            name = 'input_images')
        input_lbls = []
        for i, N_classes in enumerate(num_classes):
            input_lbls.append(
                                tf.placeholder(
                                    dtype = tf.uint8,   
                                    shape = [None, 1], # shape = [None, N_classes],
                                    name = 'input_lbls' + str(i)
                                )
                            )

        # define model model and load pre-trained model
        output_logits, endpoints, input_layer_name, output_layer_names = self._create_inference(input_images, num_classes=num_classes)
        if (use_pretrained_model):
            exclude_layers = []
            if (pretrain_exclude_input):
                exclude_layers += input_layer_name
            if (pretrain_exclude_output):
                exclude_layers += output_layer_names
            output_logits, model_vars_restored, model_vars_not_restored = self._load_pretrained_model(output_logits, pretrained_model_path, exclude_layers) #['resnet_v1_50/conv1','resnet_v1_50/logits']) #['resnet_v1_50/conv1','resnet_v1_50/logits'])
        else:
            model_vars_restored = []
            model_vars_not_restored = [value for key,value in endpoints.items()]
        # Setup loss function
        loss = self._create_losses(output_logits, input_lbls, num_classes)
        # Setup optimizer
        variables_to_optimize = None
        if (optim_vars == 'all'):
            variables_to_optimize = None
        elif (optim_vars == 'non_restored'):
            variables_to_optimize = model_vars_not_restored
        else:
            raise NotImplementedError('Value set for optim_vars not implemented. Value = ' + optim_vars)
        
        optimizer_op = self._create_optimizer(loss, variables_to_optimize=variables_to_optimize)
        
        # Setup summaries
        CMatsTrain = [CM.confusionmatrix(N_classes) for N_classes in num_classes]
        CMatsVal = [CM.confusionmatrix(N_classes) for N_classes in num_classes]

        # CMatTrain = CM.confusionmatrix(self.lbls_dim)
        # CMatVal   = CM.confusionmatrix(self.lbls_dim)
        # Setup summary dict
        # tf_accuracy = tf.placeholder(tf.float32, name='accuracy')
        # tf_precision = tf.placeholder(tf.float32, name='precision')
        # tf_recall = tf.placeholder(tf.float32, name='recall')
        # tf_F1 = tf.placeholder(tf.float32, name='F1')
        tf_loss = tf.placeholder(tf.float32, name='loss_mean')

        tf_accuracies = []
        tf_recalls = []
        tf_precisions = []
        tf_F1s = []
        tf_cs_categories = []
        for i, N_classes in enumerate(num_classes):
            tf_accuracies.append(tf.placeholder(dtype = tf.float32, name = 'Overview/Accuracy' + str(i)) )
            with tf.name_scope('output_' + str(i)):
                tf_recall, tf_chart_recall = tf_custom_summaries.class_score_mmm('Recall')
                tf_recalls.append(tf_recall)
                tf_precision, tf_chart_precision = tf_custom_summaries.class_score_mmm('Precision')
                tf_precisions.append(tf_precision)
                tf_F1, tf_chart_F1 = tf_custom_summaries.class_score_mmm('F1')
                tf_F1s.append(tf_F1)
            tf_cs_categories.append(
                                tf_custom_summaries.layout_pb2.Category(
                                    title='output' + str(i),
                                    chart=[tf_chart_F1, tf_chart_precision, tf_chart_recall]
                                )
                            )
        summary_list = tf_accuracies
        summary_dict = {'Overview/loss':         tf_loss}

        layout_summary = tf_custom_summaries.summary_lib.custom_scalar_pb(
                                tf_custom_summaries.layout_pb2.Layout(
                                    category=tf_cs_categories
                                    )
                            )
        self._create_summaries(loss, summary_dict=summary_dict, summary_list=summary_list)
        tf_summary_op = tf.summary.merge_all()
        
        # show network architecture
        # utils.show_all_variables()
        
        with tf.Session() as sess:
            
            # Initialize all model Variables.
            sess.run(tf.global_variables_initializer())                
            
            # Create Saver object for loading and storing checkpoints
            saver = tf.train.Saver()
            
            # Create Writer object for storing graph and summaries for TensorBoard
            writer_train = tf.summary.FileWriter(os.path.join(self.dir_logs,'train'), sess.graph)
            writer_validation = tf.summary.FileWriter(os.path.join(self.dir_logs,'val'), sess.graph)
            writer_train.add_summary(layout_summary)
            writer_validation.add_summary(layout_summary)
            
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
                # sess.run(tf_dataset_train_iterator.initializer)
                
                utils.show_message('Running training epoch no: {0}'.format(epoch_n), lvl=1)
                # batchCounter = 0
                
                for CMat in CMatsTrain:
                    CMat.Reset()
                # CMatTrain.Reset()
                loss_train = 0
                # while True:
                    # try:
                for batchCounter in range(math.ceil(float(dataset_sizes[0])/float(self.batch_size))):
                    interationCnt = interationCnt + 1
                    image_batch, lbl_batch, *args = sess.run(input_getBatch)

                    # Built feed dict based on list of labels
                    feed_dict = {input_lbl: np.expand_dims(lbl_batch[:,i],1) for i,input_lbl in enumerate(input_lbls)}
                    feed_dict.update({input_images:    image_batch})
                    # Perform training step
                    _, loss_out, lbl_batch_predict = sess.run(
                        [optimizer_op, loss, output_logits],
                        feed_dict=feed_dict)
                        # feed_dict={input_images:    image_batch,
                                    # input_lbls:      lbl_batch})
                    loss_train += loss_out
                    # counter =+ 1
                    # batchCounter = batchCounter + 1
                    
                    # Calculate confusion matrix for all outputs
                    for i,CMat in enumerate(CMatsTrain):
                        lbl_idx = lbl_batch[:,i] #np.squeeze(np.argmax(lbl_batch, axis=1))
                        lbl_idx_predict = np.squeeze(np.argmax(lbl_batch_predict[i], axis=3))
                        CMat.Append(lbl_idx,lbl_idx_predict)

                    # lbl_idx = lbl_batch[:,0] #np.squeeze(np.argmax(lbl_batch, axis=1))
                    # lbl_idx_predict = np.squeeze(np.argmax(lbl_batch_predict[0], axis=3))

                    # CMatTrain.Append(lbl_idx,lbl_idx_predict)

                    # print('T' + str(epoch_n) + ' ' + str(batchCounter)  + ' ' + str(lbl_idx) + ' ' + str(lbl_idx_predict) + ' ' + str(loss_out))
                    # print('T' + '{:d}'.format(epoch_n) + ' ' + '{:>4d}'.format(batchCounter)  + ' ' + '{:>9.3f}'.format(loss_out) + ' ' + ' '.join(['{:>5.3f}'.format(CMat.accuracy()) for CMat in CMatsTrain]))

                    # TODO: overwrite previous output. stdout.flush?
                    # print('T' + '{:d}'.format(epoch_n) + ' ' + '{:>4d}'.format(batchCounter)  + ' Loss: ' + '{:>7.3g}'.format(loss_out) + ' Acc(s): ' + '  '.join(['{:>5.3f}'.format(CMat.accuracy()) for CMat in CMatsTrain]))
                    self._show_progress('T', epoch_n, batchCounter, math.ceil(float(dataset_sizes[0])/float(self.batch_size)), loss_out, CMatsTrain)
                        
                    # except tf.errors.OutOfRangeError:

                # Print accumulated confusion matricx for each output
                for i, CMat in enumerate(CMatsTrain):
                    CMat.Save(os.path.join(self.dir_logs, 'ConfMat_Train_output' + '{:02d}'.format(i) + '.csv'),'csv')
                    print(CMat)
                    print(CMat.accuracy())
                
                # Extract training parameters and store in log-file
                # accuracy = CMatTrain.accuracy()
                # precision = [0 if np.isnan(x) else x for x in CMatTrain.precision()]
                # recall = [0 if np.isnan(x) else x for x in CMatTrain.recall()]
                # F1 = [0 if np.isnan(x) else x for x in CMatTrain.fScore(beta=1)]
                loss_train = loss_train/batchCounter

                feed_dict_summary = {tf_acc: CMat.accuracy() for tf_acc, CMat in zip(tf_accuracies,CMatsTrain)}
                # feed_dict_summary.update({tf_rec: np.mean([0 if np.isnan(x) else x for x in CMat.recall()]) for tf_rec, CMat in zip(tf_recalls,CMatsTrain)})
                # feed_dict_summary.update({tf_pre: np.mean([0 if np.isnan(x) else x for x in CMat.precision()]) for tf_pre, CMat in zip(tf_precisions,CMatsTrain)})
                # feed_dict_summary.update({tf_f1:  np.mean([0 if np.isnan(x) else x for x in CMat.fScore(beta=1)]) for tf_f1, CMat in zip(tf_F1s,CMatsTrain)})
                feed_dict_summary.update({tf_rec: [0 if np.isnan(x) else x for x in CMat.recall()] for tf_rec, CMat in zip(tf_recalls,CMatsTrain)})
                feed_dict_summary.update({tf_pre: [0 if np.isnan(x) else x for x in CMat.precision()] for tf_pre, CMat in zip(tf_precisions,CMatsTrain)})
                feed_dict_summary.update({tf_f1:  [0 if np.isnan(x) else x for x in CMat.fScore(beta=1)] for tf_f1, CMat in zip(tf_F1s,CMatsTrain)})
                # feed_dict_summary.update({tf_recall_test: [0 if np.isnan(x) else x for x in CMatsTrain[0].recall()]})
                # feed_dict_summary.update({ tf_accuracy:    accuracy,
                #                             tf_precision:   np.mean(precision),
                #                             tf_recall:      np.mean(recall),
                #                             tf_F1:          np.mean(F1)})
                feed_dict_summary.update({tf_loss: loss_train})

                summaries = sess.run(tf_summary_op, 
                                    feed_dict=feed_dict_summary)
                writer_train.add_summary(summaries, global_step=epoch_n)

                ########
                # Validation
                ########
                if (tf_dataset_val is not None):
                    utils.show_message('Running validation epoch no: {0}'.format(epoch_n),lvl=1)
                    # Initiate or Re-initiate iterator
                    # sess.run(tf_dataset_val_iterator.initializer)
                    # batchCounter = 0
                    for CMat in CMatsVal:
                        CMat.Reset()
                    loss_val = 0
                    for batchCounter in range(math.ceil(float(dataset_sizes[1])/float(self.batch_size))):
                    # while True:
                        # try:
                        image_batch, lbl_batch, *args = sess.run(tf_input_getBatch_val)

                        # Built feed dict based on list of labels
                        feed_dict = {input_lbl: np.expand_dims(lbl_batch[:,i],1) for i,input_lbl in enumerate(input_lbls)}
                        feed_dict.update({input_images:    image_batch})

                        # Perform evaluation step
                        lbl_batch_predict, loss_out = sess.run(
                                                            [output_logits, loss],
                                                            feed_dict=feed_dict
                                                        )

                        # counter =+ 1
                        # batchCounter = batchCounter + 1
                        # lbl_idx = np.squeeze(np.argmax(lbl_batch, axis=1))
                        # lbl_idx_predict = np.squeeze(np.argmax(lbl_batch_predict, axis=3))
                        
                        # Calculate confusion matrix for all outputs
                        for i,CMat in enumerate(CMatsVal):
                            lbl_idx = lbl_batch[:,i] #np.squeeze(np.argmax(lbl_batch, axis=1))
                            lbl_idx_predict = np.squeeze(np.argmax(lbl_batch_predict[i], axis=3))
                            CMat.Append(lbl_idx,lbl_idx_predict)

                        # lbl_idx = lbl_batch[:,0] #np.squeeze(np.argmax(lbl_batch, axis=1))
                        # lbl_idx_predict = np.squeeze(np.argmax(lbl_batch_predict[0], axis=3))

                        loss_val += loss_out

                        # CMatVal.Append(lbl_idx,lbl_idx_predict)

                        # print('V' + str(epoch_n) + ' ' + str(batchCounter)  + ' ' + str(lbl_idx) + ' ' + str(lbl_idx_predict) + ' ' + str(loss_out))

                        # print('V' + str(epoch_n) + ' ' + str(batchCounter)  + ' ' + str(loss_out) + ' '.join([str(CMat.accuracy()) for CMat in CMatsVal]))

                        # TODO: overwrite previous output. stdout.flush?
                        # print('V' + '{:d}'.format(epoch_n) + ' ' + '{:>4d}'.format(batchCounter)  + ' Loss: ' + '{:>7.3g}'.format(loss_out) + ' Acc(s): ' + '  '.join(['{:>5.3f}'.format(CMat.accuracy()) for CMat in CMatsVal]))
                            
                        # except tf.errors.OutOfRangeError:
                    for i, CMat in enumerate(CMatsVal):
                        CMat.Save('ConfMat_Val_output' + '{:02d}'.format(i) + '.csv','csv')
                        CMat.Save(os.path.join(self.dir_logs, 'ConfMat_Val_output' + '{:02d}'.format(i) + '.csv'),'csv')
                        print(CMat)
                        print(CMat.accuracy())
                    # Do some evaluation after each Epoch
                    # accuracy = CMatVal.accuracy()
                    # precision = [0 if np.isnan(x) else x for x in CMatVal.precision()]
                    # recall = [0 if np.isnan(x) else x for x in CMatVal.recall()]
                    # F1 = [0 if np.isnan(x) else x for x in CMatVal.fScore(beta=1)]
                    loss_val = loss_val/batchCounter

                    feed_dict_summary = {tf_acc: CMat.accuracy() for tf_acc, CMat in zip(tf_accuracies,CMatsVal)}
                    # feed_dict_summary.update({tf_rec: np.mean([0 if np.isnan(x) else x for x in CMat.recall()]) for tf_rec, CMat in zip(tf_recalls,CMatsVal)})
                    # feed_dict_summary.update({tf_pre: np.mean([0 if np.isnan(x) else x for x in CMat.precision()]) for tf_pre, CMat in zip(tf_precisions,CMatsVal)})
                    # feed_dict_summary.update({tf_f1:  np.mean([0 if np.isnan(x) else x for x in CMat.fScore(beta=1)]) for tf_f1, CMat in zip(tf_F1s,CMatsVal)})
                    feed_dict_summary.update({tf_rec: [0 if np.isnan(x) else x for x in CMat.recall()] for tf_rec, CMat in zip(tf_recalls,CMatsVal)})
                    feed_dict_summary.update({tf_pre: [0 if np.isnan(x) else x for x in CMat.precision()] for tf_pre, CMat in zip(tf_precisions,CMatsVal)})
                    feed_dict_summary.update({tf_f1:  [0 if np.isnan(x) else x for x in CMat.fScore(beta=1)] for tf_f1, CMat in zip(tf_F1s,CMatsVal)})
                    # feed_dict_summary.update({tf_recall_test: [0 if np.isnan(x) else x for x in CMatsVal[0].recall()]})
                    # feed_dict_summary.update({ tf_accuracy:    accuracy,
                    #                             tf_precision:   np.mean(precision),
                    #                             tf_recall:      np.mean(recall),
                    #                             tf_F1:          np.mean(F1)})
                    feed_dict_summary.update({tf_loss: loss_val})

                    summaries = sess.run(tf_summary_op, 
                                        feed_dict=feed_dict_summary)
                                                    # { tf_accuracy:    accuracy,
                                                    # tf_precision:   np.mean(precision),
                                                    # tf_recall:      np.mean(recall),
                                                    # tf_F1:          np.mean(F1),
                                                    # tf_loss:        loss_val})
                                                    # tf_accuracy:    np.asarray(np.reshape(accuracy,(1, 1, -1,1))*255,dtype=np.uint8),
                                                    # tf_precision:   np.asarray(np.reshape(precision,(1, 1, -1,1))*255,dtype=np.uint8),
                                                    # tf_recall:      np.asarray(np.reshape(recall,(1, 1, -1,1))*255,dtype=np.uint8),
                                                    # tf_F1:          np.asarray(np.reshape(F1,(1, 1, -1,1))*255,dtype=np.uint8)})
                    writer_validation.add_summary(summaries, global_step=epoch_n)
                            # break
                
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
    

    # def _preProcessData(self, image_proto, lbl_proto, class_proto, height_proto, width_proto, channels_proto, origin_proto):
    #     """ Local preprocessing of data from dataset
    #     also used to select which elements to parse onto the model
    #     Args:
    #       all outputs of util_data.decode_image

    #     Returns:
    #     """

        # Seeds only
        # Perform random rotation
        # rotations = tf.random.uniform([1,],minval=0,maxval=6.28)
        # image_proto = tf.contrib.image.rotate(image_proto, angles=rotations)

        # Perform resize to desired size
        # image_proto = tf.image.resize_image_with_crop_or_pad(image_proto, target_height=self.image_dims[0], target_width=self.image_dims[1])

        # One-hot encode labels
        # lbl = tf.one_hot(lbl_proto, self.lbls_dim)

        # PSD only
        # pad_to_size()
        # image_proto_shape = tf.shape(image_proto)
        # image_height = tf.cond(tf.equal(tf.rank(image_proto),4),lambda: image_proto_shape[1], lambda: image_proto_shape[0])
        # image_width = tf.cond(tf.equal(tf.rank(image_proto),4),lambda: image_proto_shape[2],lambda: image_proto_shape[1])
        # image_proto = tf.image.pad_to_bounding_box(image_proto,
        #                                             offset_height=tf.floordiv(400-image_height,2),
        #                                             offset_width=tf.floordiv(400-image_width,2),
        #                                             target_height=400,
        #                                             target_width=400)

        # preprocessing = self.preprocess_factory()

        # params_pad = {'target_height': 400, 'target_width': 400}
        # image_proto, lbl_proto, *_ = preprocessing.pad_to_size(params_pad, image_proto, lbl_proto, class_proto, height_proto, width_proto, channels_proto, origin_proto)

        # params_rand_rot = {} # Leave empty to use default values
        # image_proto, lbl_proto, *_ = preprocessing.random_rotation(params_rand_rot, image_proto, lbl_proto, class_proto, height_proto, width_proto, channels_proto, origin_proto)

        # param_dict_resize = {'target_height': 128, 'target_width': 128}
        # image_proto, lbl_proto, *_ = preprocessing.resize(param_dict_resize, image_proto, lbl_proto, class_proto, height_proto, width_proto, channels_proto, origin_proto)

        # params = [(preprocessing.pad_to_size, params_pad), (preprocessing.random_rotation, params_rand_rot), (preprocessing.resize, param_dict_resize)]

        # preprocessing.built_pipe(params, image_proto, lbl_proto, class_proto, height_proto, width_proto, channels_proto, origin_proto)

        # return image_proto, lbl_proto, class_proto, height_proto, width_proto, channels_proto, origin_proto
