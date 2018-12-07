import os
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import shlex
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import vgg

a = slim.get_model_variables('vgg_16')

init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'vgg_16.ckpt'),
        slim.get_model_variables('vgg_16'))