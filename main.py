"""
This file is used to run the project.
Notes:
- The structure of this file (and the entire project in general) is made with emphasis on flexibility for research
purposes, and the pipelining is done in a python file such that newcomers can easily use and understand the code.
- Remember that relative paths in Python are always relative to the current working directory.

Hence, if you look at the functions in make_dataset.py, the file paths are relative to the path of
this file (main.py)
"""

__author__ = "Simon Leminen Madsen"
__email__ = "slm@eng.au.dk"

import os
import GPUtil
import argparse
import datetime

import src.utils as utils
from src.data import dataset_manager
from src.models.BasicModel import BasicModel
from src.models.logreg_example import logreg_example
from src.models.VGG import VGG
from src.models.ResNet import ResNet
from src.visualization import visualize

DEVICE_ID_LIST = GPUtil.getFirstAvailable(attempts = 720, interval = 1, maxMemory=1.0, maxLoad=1.0)
GPUtil.showUtilization()
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID_LIST[0])

"""parsing and configuration"""
def parse_args():
    
# ----------------------------------------------------------------------------------------------------------------------
# Define default pipeline
# ----------------------------------------------------------------------------------------------------------------------

    desc = "Pipeline for running Tensorflow implementation of infoGAN"
    parser = argparse.ArgumentParser(description=desc)
    
    parser.add_argument('--make_dataset', 
                        action='store_true', 
                        help = 'Fetch dataset from remote source into /data/raw/. Or generate raw dataset [Defaults to False if argument is omitted]')
    
    parser.add_argument('--process_dataset', 
                        action='store_true', 
                        help = 'Run preprocessing of raw data. [Defaults to False if argument is omitted]')

    parser.add_argument('--train_model', 
                        action='store_true', 
                        help = 'Run configuration and training network [Defaults to False if argument is omitted]')
    
    parser.add_argument('--evaluate_model', 
                        action='store_true', 
                        help = 'Run evaluation of the model by computing and visualizing the results [Defaults to False if argument is omitted]')
    
    parser.add_argument('--visualize', 
                        action='store_true', 
                        help = 'Run visualization of results [Defaults to False if argument is omitted]')
    
# ----------------------------------------------------------------------------------------------------------------------
# Define the arguments used in the entire pipeline
# ----------------------------------------------------------------------------------------------------------------------

    parser.add_argument('--model', 
                        type=str, 
                        default='BasicModel', 
                        choices=['BasicModel',
                                 'LogReg_example',
                                 'VGG',
                                 'ResNet'],
                        #required = True,
                        help='The name of the network model')

    parser.add_argument('--dataset', 
                        type=str, default='custom_jpeg', 
                        choices=['MNIST',
                                 'PSD_Nonsegmented',
                                 'PSD_Segmented',
                                 'custom_jpeg',
                                 'seeds_all',
                                 'barley',
                                 'barley_abnormal',
                                 'barley_d0',
                                 'barley_next',
                                 'okra',
                                 'okra_abnormal',
                                 'okra_next',
                                 'okra_d0',
                                 'OSD_Ra1',
                                 'OSD_Ra1W',
                                 'OSD_Weeds',
                                 'OSD_Ra',
                                 'OSD',
                                 'OSD_new_wc',
                                 'OSD_Weeds_new_wc',
                                 'OSD_seed',
                                 'OSD_Weeds_seed',
                                 'OSD_seed_new_wc',
                                 'OSD_Weeds_seed_new_wc',
                                 'OSD_wc_repeat',
                                 'OSD_seed_new_focus'],
                        #required = True,
                        help='The name of dataset')  
    
# ----------------------------------------------------------------------------------------------------------------------
# Define the arguments for the training
# ----------------------------------------------------------------------------------------------------------------------

    parser.add_argument('--id',
                        type= str,
                        default = None,
                        help = 'Optional ID, to distinguise experiments')

    parser.add_argument('--hparams',
                        type=str, default = '',
                        help='CLI arguments for the model wrapped in a string')
    
    parser.add_argument('--preprocess',
                        type=str, default = '',
                        help='Specify the preprocessing steps with assossiated parameters used during training step.')
    parser.add_argument('--preprocess_eval',
                        type=str, default = '',
                        help='Specify the preprocessing steps with assossiated parameters used during validation and testing steps.')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    
    # Assert if training parameters are provided, when training is selected
#    if args.train_model:
#        try:
#            assert args.hparams is ~None
#        except:
#            print('hparams not provided for training')
#            exit()
    print(args)
    return args


"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # Set tensorflow log level (0 = all, 1 >= no INFO, 2 >= no WARNING, 3 >= no ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # or any {'0', '1', '2', '3'}
    
    # Make dataset
    if args.make_dataset:
        utils.show_message('Fetching raw dataset: {0}'.format(args.dataset), lvl = 1)
        dataset_manager.make_dataset(args.dataset)
         
    # Make dataset
    if args.process_dataset:
        utils.show_message('Processing raw dataset: {0}'.format(args.dataset), lvl = 1)
        dataset_manager.process_dataset(args.dataset)

        
    # Build and train model
    if args.train_model:
        utils.show_message('Configuring and Training Network: {0}'.format(args.model), lvl = 1)
        
        if args.model == 'BasicModel':
            model = BasicModel(
                dataset = args.dataset,
                id = args.id)
            model.train(hparams_string = args.hparams)

        elif args.model == 'LogReg_example':
            model = logreg_example(
                dataset = args.dataset,
                id = args.id)
            model.train(hparams_string = args.hparams)

        elif args.model == 'VGG':
            model = VGG(
                dataset = args.dataset,
                id = args.id)
            model.train(hparams_string = args.hparams)

        elif args.model == 'ResNet':
            model = ResNet(dataset = args.dataset,
                id = args.id)
            model.train(hparams_string = args.hparams, preprocessing_params=args.preprocess, preprocessing_eval_params=args.preprocess_eval)
            
    

    # Evaluate model
    if args.evaluate_model:
        utils.show_message('Evaluating Network: {0}'.format(args.model), lvl = 1)

        if args.model == 'BasicModel':
            model = BasicModel(
                dataset = args.dataset,
                id = args.id)
            model.evaluate(hparams_string = args.hparams)

        elif args.model == 'LogReg_example':
            model = logreg_example(
                dataset = args.dataset,
                id = args.id)
            model.evaluate(hparams_string = args.hparams)

        elif args.model == 'ResNet':
            model = ResNet(dataset = args.dataset,
                id = args.id)
            model.evaluate(hparams_string = args.hparams, preprocessing_params=args.preprocess)


    # Visualize results
    if args.visualize:
        print('Visualizing Results')
        #################################
        ####### To Be Implemented #######
        #################################
        raise NotImplementedError('Visualization has not yet been implemented.')
    

if __name__ == '__main__':
    main()
