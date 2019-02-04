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

def dataset_parser_group(parser):
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
    return parser_group_dataset

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
    parser.add_argument('--global_pool',
                        type=str2bool, default=True,
                        help='Flag specifies if mean pooling should be performed before the last convolution layer of ResNet. Default: %(default)s')

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

    # Add dataset parameters
    parser_group_dataset = dataset_parser_group(parser)


    ## add more model parameters to enable configuration from terminal
    
    return parser.parse_args(shlex.split(hparams_string))


def hparams_parser_evaluate(hparams_string):
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch_no', 
                        type=int,
                        default=None, 
                        help='Epoch no to reload')

    parser.add_argument('--batch_size', 
                        type=int, default='10',
                        help='Number of samples in each batch')

    # Add dataset parameters
    parser_group_dataset = dataset_parser_group(parser)

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
        
       
    def _create_inference(self, inputs, num_classes, is_training = True, global_pool=True, dropout_keep_prob = 0.5):
        """ Define the inference model for the network
        Args:
    
        Returns:
        """
        utils.show_message('Create model inference', lvl=1)
        print('Model: ' + self.model_version)

        if self.model_version == 'ResNet50':
            logits, endpoints = resnet_v1.resnet_v1_50(inputs, num_classes, is_training=is_training, global_pool=global_pool, spatial_squeeze=False)
            input_layer_name = ['resnet_v1_50/conv1']
            output_layer_names = [ep for ep in endpoints if ('logits' in ep)]

        elif self.model_version == 'ResNet101':
            logits, endpoints = resnet_v1.resnet_v1_101(inputs, num_classes, is_training=is_training, global_pool=global_pool, spatial_squeeze=False)
            input_layer_name = ['resnet_v1_101/conv1']
            output_layer_names = [ep for ep in endpoints if ('logits' in ep)]
        
        elif self.model_version == 'ResNet152':
            logits, endpoints = resnet_v1.resnet_v1_152(inputs, num_classes, is_training=is_training, global_pool=global_pool, spatial_squeeze=False)
            input_layer_name = ['resnet_v1_152/conv1']
            output_layer_names = [ep for ep in endpoints if ('logits' in ep)]

        elif self.model_version == 'ResNet200':
            logits, endpoints = resnet_v1.resnet_v1_200(inputs, num_classes, is_training=is_training, global_pool=global_pool, spatial_squeeze=False)
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
        batch_counter_len = math.ceil(math.log10(batch_max))
        output_string = tag + '{:d}'.format(epoch) + ' ' + '{:>d}'.format(batch_counter).rjust(batch_counter_len) + '/' + '{:>d}'.format(batch_max).rjust(batch_counter_len)  + ' Loss: ' + '{:>7.3g}'.format(loss) + ' Acc(s): ' + '  '.join(['{:>5.3f}'.format(CMat.accuracy()) for CMat in CMats])
        sys.stdout.write('\r'+output_string)
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
            tf_dataset_train = tf_dataset_train.map(preprocessing.pipe)
            tf_dataset_train = tf_dataset_train.batch(batch_size = self.batch_size, drop_remainder=False)
            tf_dataset_train = tf_dataset_train.repeat(count=-1) # -1 --> repeat indefinitely
            tf_dataset_train = tf_dataset_train.prefetch(buffer_size=3)
            tf_dataset_train_iterator = tf_dataset_train.make_one_shot_iterator()
            input_getBatch = tf_dataset_train_iterator.get_next()

        with tf.name_scope('Validation_dataset'):
            tf_dataset_val = tf_dataset_list[1]
            if (tf_dataset_val is not None):
                tf_dataset_val = tf_dataset_val.map(DS._decode_from_TFexample)
                tf_dataset_val = tf_dataset_val.map(preprocessing.pipe)
                tf_dataset_val = tf_dataset_val.batch(batch_size = self.batch_size, drop_remainder=False)
                tf_dataset_val = tf_dataset_val.repeat(count=-1) # -1 --> repeat indefinitely
                tf_dataset_val = tf_dataset_val.prefetch(buffer_size=3)
                tf_dataset_val_iterator = tf_dataset_val.make_one_shot_iterator()
                tf_input_getBatch_val = tf_dataset_val_iterator.get_next()

        # Define input and output layers
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
        output_logits, endpoints, input_layer_name, output_layer_names = self._create_inference(input_images, num_classes=num_classes, global_pool=args_train.global_pool)
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
            
            # Do training loops
            for epoch_n in range(epoch_start, self.epoch_max):
                
                #################
                # Training step #
                #################
                utils.show_message('Running training epoch no: {0}'.format(epoch_n), lvl=1)
                # Reset confusion matrices and accumulated loss
                for CMat in CMatsTrain:
                    CMat.Reset()
                loss_train = 0
                 # Loop through all batches of examples
                for batchCounter in range(math.ceil(float(dataset_sizes[0])/float(self.batch_size))):
                    # Grab an image and label batch from the validation set
                    image_batch, lbl_batch, *args = sess.run(input_getBatch)
                    # Built feed dict based on list of labels
                    feed_dict = {input_lbl: np.expand_dims(lbl_batch[:,i],1) for i,input_lbl in enumerate(input_lbls)}
                    feed_dict.update({input_images:    image_batch})
                    # Perform training step
                    _, loss_out, lbl_batch_predict = sess.run(
                        [optimizer_op, loss, output_logits],
                        feed_dict=feed_dict)
                    loss_train += loss_out
                    # Store results from training step
                    # Calculate confusion matrix for all outputs
                    for i,CMat in enumerate(CMatsTrain):
                        lbl_idx = lbl_batch[:,i]
                        lbl_idx_predict = np.squeeze(np.argmax(lbl_batch_predict[i], axis=3))
                        CMat.Append(lbl_idx,lbl_idx_predict)
                    # Show progress in stdout
                    self._show_progress('TR', epoch_n, batchCounter, math.ceil(float(dataset_sizes[0])/float(self.batch_size))-1, loss_out, CMatsTrain)

                # Print accumulated confusion matricx for each output
                print('\n')
                for i, CMat in enumerate(CMatsTrain):
                    CMat.Save(os.path.join(self.dir_logs, 'ConfMat_Train_output' + '{:02d}'.format(i) + '.csv'),'csv')
                    print(CMat)
                
                # Create fill in summaries for training log
                feed_dict_summary = {tf_acc: CMat.accuracy() for tf_acc, CMat in zip(tf_accuracies,CMatsTrain)}
                feed_dict_summary.update({tf_rec: [0 if np.isnan(x) else x for x in CMat.recall()] for tf_rec, CMat in zip(tf_recalls,CMatsTrain)})
                feed_dict_summary.update({tf_pre: [0 if np.isnan(x) else x for x in CMat.precision()] for tf_pre, CMat in zip(tf_precisions,CMatsTrain)})
                feed_dict_summary.update({tf_f1:  [0 if np.isnan(x) else x for x in CMat.fScore(beta=1)] for tf_f1, CMat in zip(tf_F1s,CMatsTrain)})
                loss_train = loss_train/batchCounter
                feed_dict_summary.update({tf_loss: loss_train})
                summaries = sess.run(tf_summary_op, 
                                    feed_dict=feed_dict_summary)
                # Write summaries to training log
                writer_train.add_summary(summaries, global_step=epoch_n)

                ###################
                # Validation step #
                ###################

                if (tf_dataset_val is not None): # Skip validation step, if there is no validation dataset
                    utils.show_message('Running validation epoch no: {0}'.format(epoch_n),lvl=1)
                    # Reset confusion matrices and accumulated loss
                    for CMat in CMatsVal:
                        CMat.Reset()
                    loss_val = 0
                    # Loop through all batches of examples
                    for batchCounter in range(math.ceil(float(dataset_sizes[1])/float(self.batch_size))):
                        # Grab an image and label batch from the validation set
                        image_batch, lbl_batch, *args = sess.run(tf_input_getBatch_val)
                        # Built feed dict based on list of labels
                        feed_dict = {input_lbl: np.expand_dims(lbl_batch[:,i],1) for i,input_lbl in enumerate(input_lbls)}
                        feed_dict.update({input_images:    image_batch})
                        # Perform evaluation step
                        lbl_batch_predict, loss_out = sess.run(
                                                            [output_logits, loss],
                                                            feed_dict=feed_dict
                                                        )
                        # Store results from evaluation step
                        # Calculate confusion matrix for all outputs
                        for i,CMat in enumerate(CMatsVal):
                            lbl_idx = lbl_batch[:,i] #np.squeeze(np.argmax(lbl_batch, axis=1))
                            lbl_idx_predict = np.squeeze(np.argmax(lbl_batch_predict[i], axis=3))
                            CMat.Append(lbl_idx,lbl_idx_predict)
                        loss_val += loss_out
                        # Show progress in stdout
                        self._show_progress('VA', epoch_n, batchCounter, math.ceil(float(dataset_sizes[1])/float(self.batch_size))-1, loss_out, CMatsVal)
                    
                    # Print confusion matrix for each output
                    print('\n')
                    for i, CMat in enumerate(CMatsVal):
                        CMat.Save(os.path.join(self.dir_logs, 'ConfMat_Val_output' + '{:02d}'.format(i) + '.csv'),'csv') # Save confusion matrix
                        print(CMat)

                    # Create fill in summaries for validation log
                    feed_dict_summary = {tf_acc: CMat.accuracy() for tf_acc, CMat in zip(tf_accuracies,CMatsVal)}
                    feed_dict_summary.update({tf_rec: [0 if np.isnan(x) else x for x in CMat.recall()] for tf_rec, CMat in zip(tf_recalls,CMatsVal)})
                    feed_dict_summary.update({tf_pre: [0 if np.isnan(x) else x for x in CMat.precision()] for tf_pre, CMat in zip(tf_precisions,CMatsVal)})
                    feed_dict_summary.update({tf_f1:  [0 if np.isnan(x) else x for x in CMat.fScore(beta=1)] for tf_f1, CMat in zip(tf_F1s,CMatsVal)})
                    loss_val = loss_val/batchCounter
                    feed_dict_summary.update({tf_loss: loss_val})
                    summaries = sess.run(tf_summary_op, 
                                        feed_dict=feed_dict_summary)
                    # Write summaries to validation log
                    writer_validation.add_summary(summaries, global_step=epoch_n)
                
                # Save checkpoint for this epoch
                if epoch_n % 1 == 0:
                    saver.save(sess,os.path.join(self.dir_checkpoints, self.model + '.model'), global_step=epoch_n)
                
            
    
    def evaluate(self, hparams_string, preprocessing_params=''):
        """ Run prediction of the network
        Args:
    
        Returns:
        """
        args_evaluate = hparams_parser_evaluate(hparams_string)


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
        tf_dataset_list, dataset_sizes = DS.get_dataset_list(shuffle_before_split=args_evaluate.shuffle_before_split,
                                                            shuffle_seed=args_evaluate.shuffle_seed,
                                                            group_before_split=args_evaluate.group_before_split,
                                                            validation_method=args_evaluate.validation_method,
                                                            holdout_split=args_evaluate.holdout_split,
                                                            cross_folds=10,
                                                            cross_val_fold=None,
                                                            cross_test_fold=0,
                                                            shard_val=args_evaluate.shard_val,
                                                            shard_test=args_evaluate.shard_test,
                                                            stratify_training_set=args_evaluate.stratify_training_set)

        class_dicts = DS.get_class_dicts()
        num_classes = [len(class_dict) for class_dict in class_dicts]

        preprocessing = preprocess_factory.preprocess_factory()
        if not (preprocessing_params == ''):
            # Setup preprocessing pipeline
            preprocessing.prep_pipe_from_string(preprocessing_params)

        with tf.name_scope('Test_dataset'):
            tf_dataset_test = tf_dataset_list[2]
            if (tf_dataset_test is not None):
                tf_dataset_test = tf_dataset_test.map(DS._decode_from_TFexample)
                tf_dataset_test = tf_dataset_test.map(preprocessing.pipe)
                tf_dataset_test = tf_dataset_test.batch(batch_size = args_evaluate.batch_size, drop_remainder=False)
                tf_dataset_test = tf_dataset_test.prefetch(buffer_size=3)
                tf_dataset_test_iterator = tf_dataset_test.make_one_shot_iterator()
                tf_input_getBatch_test = tf_dataset_test_iterator.get_next()

        CMatsTest = [CM.confusionmatrix(N_classes) for N_classes in num_classes]

        with tf.Session() as tf_session:
   
            # Locate checkpoints and load the latest metagraph and checkpoint
            ckpt = tf.train.get_checkpoint_state(self.dir_checkpoints)
            saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
            saver.restore(tf_session, tf.train.latest_checkpoint(self.dir_checkpoints))

            # Grab input and output tensors
            graph = tf.get_default_graph()
            input_images = graph.get_tensor_by_name('input_images:0')
            input_lbls = []
            output_logits = []
            for i, N_classes in enumerate(num_classes):
                input_lbls.append(graph.get_tensor_by_name('input_lbls' + str(i) + ':0'))
                output_logits.append(graph.get_tensor_by_name('resnet_v1_101/logits' + str(i) + '/BiasAdd:0'))
                # output_logits.append(graph.get_tensor_by_name('logits' + str(i) + ':0'))

            # results_list_file = os.path.join(self.dir_results, 'results_list.csv')
            # fob_results_list = open(results_list_file,'w')

            # Reset confusion matrices and accumulated loss
            for CMat in CMatsTest:
                CMat.Reset()
            loss_acc = 0
            # Loop through all batches of examples
            for batchCounter in range(math.ceil(float(dataset_sizes[2])/float(args_evaluate.batch_size))):
                # Grab an image and label batch from the validation set
                image_batch, lbl_batch, *args = tf_session.run(tf_input_getBatch_test)
                # Built feed dict based on list of labels
                # feed_dict = {input_lbl: np.expand_dims(lbl_batch[:,i],1) for i,input_lbl in enumerate(input_lbls)}
                # feed_dict.update({input_images:    image_batch})
                feed_dict = {input_images:    image_batch}
                # Perform evaluation step
                lbl_batch_predict = tf_session.run([output_logits],
                                                    feed_dict=feed_dict
                                                )
                # Store results from evaluation step
                # Calculate confusion matrix for all outputs
                for i,CMat in enumerate(CMatsTest):
                    lbl_idx = lbl_batch[:,i]
                    lbl_idx_predict = np.squeeze(np.argmax(lbl_batch_predict[i][0], axis=3))
                    CMat.Append(lbl_idx,lbl_idx_predict)
                # Show progress in stdout
                self._show_progress('TE', 0, batchCounter, math.ceil(float(dataset_sizes[2])/float(args_evaluate.batch_size))-1, np.nan, CMatsTest)
            
            # Print confusion matrix for each output
            print('\n')
            for i, CMat in enumerate(CMatsTest):
                CMat.Save(os.path.join(self.dir_results, 'ConfMat_Test_output' + '{:02d}'.format(i) + '.csv'),'csv') # Save confusion matrix
                print(CMat)
            
        

        
