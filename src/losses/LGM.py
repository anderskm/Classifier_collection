import colorsys
import io
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.stats import multivariate_normal
import scipy.special
import tensorflow as tf

class LGM:
    # Based on / inspired by PyTorch implementation:
    # https://github.com/YirongMao/softmax_variants/blob/master/model_utils.py#L138
    def __init__(self, input_vector, num_classes, num_feats=None, lmbda=0.1, alpha=0.0, centers=None, log_covars=None, class_names=None, weight_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.0001), bias_initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0)):
        
        self.num_classes = num_classes

        if num_feats is None:
            # Set number of features such that there are one dimension for each combination of two classes. Limited to 46 classes --> 1035 features.
            num_feats = np.int64(scipy.special.comb(np.minimum(num_classes, 46), 2))
            print('LGM: Number of features not specified. Set to ' + str(num_feats))
        self.num_feats = num_feats

        self.lmbda = lmbda
        self.alpha = alpha

        with tf.name_scope('LGM_Embedded_space'):
            self._LGM_weights = tf.Variable(initial_value=weight_initializer(shape=(1,1,input_vector.get_shape().as_list()[-1], self.num_feats)), name='weights')
            # LGM_weights = tf.Variable(initial_value=tf.initializers.he_normal()(shape=(1,1,endpoints['global_pool'].get_shape().as_list()[-1],num_features)))
            self._LGM_biases = tf.Variable(initial_value=bias_initializer(shape=(1,1,1,self.num_feats)), name='bias')
            # LGM_biases = tf.Variable(initial_value=tf.initializers.he_normal()(shape=(1,1,1,num_features)))
            self._LGM_space = tf.nn.conv2d(input_vector, self._LGM_weights, strides=[1,1,1,1], padding='SAME')
            self._LGM_space = self._LGM_space + self._LGM_biases

        with tf.name_scope('LGM_Distributions'):
            if centers is None:
                self.centers = tf.Variable(initial_value=tf.random_normal_initializer(mean=0.0, stddev=5.0)(shape=(num_feats, num_classes)), name='centers')
                # self.centers = tf.Variable(initial_value=tf.initializers.he_normal()(shape=(num_feats, num_classes)))
            else:
                self.centers = centers

            # Assuming diagonal covariance matrix
            if log_covars is None:
                self.log_covars = tf.Variable(initial_value=tf.zeros_initializer()(shape=(num_feats, num_classes)), name='log_covars')
                # self.log_covars = tf.Variable(initial_value=tf.random_normal_initializer(mean=0.0, stddev=1.0)(shape=(num_feats, num_classes)))
            else:
                self.log_covars = log_covars

        class_hues = [float(i)/self.num_classes for i in range(self.num_classes)]
        self.class_colors = [colorsys.hsv_to_rgb(class_hue, 1.0, 1.0) for class_hue in class_hues]
        self.color_map = LinearSegmentedColormap.from_list('class_colors', self.class_colors, N=self.num_classes)

        if class_names is None:
            class_names = ['{:d}'.format(i) for i in range(self.num_classes)]
        self.class_names = class_names

    def apply(self, feat_vec, labels):
        with tf.name_scope('LGM_Loss'):
        # Feat_vec : B*1*1*num_feats
        # Labels   : B*1*1*1
            with tf.name_scope('One_hot_labels'):
                labels_one_hot = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.one_hot(tf.squeeze(labels),depth=self.num_classes),1),1),1)

            self.D = self._D(self._LGM_space) # (eq. 18, but for all i and k)

            batch_size = tf.cast(tf.shape(self._LGM_space)[0], tf.float32)

            self.Lcls = self._classification_loss(self.D, labels_one_hot)/batch_size
            self.Llkd = self._likelihood_regularization(self.D, labels_one_hot)/batch_size

            with tf.name_scope('Loss_sum'):
                return (self.Lcls + self.lmbda * self.Llkd)

    def figure_to_array(self, figure):
        figure.canvas.draw()
        X = np.frombuffer(figure.canvas.renderer.buffer_rgba(), dtype=np.uint8)
        X = X.reshape(figure.canvas.get_width_height()[::-1] + (4,))
        plt.close(figure)
        return np.expand_dims(X, axis=0)

    def plot_feat_vec(self, feat_vec, labels):
        
        figure = plt.figure(figsize=(8,8))
        figure.tight_layout(pad=0)
        for i in range(self.num_classes):
            _feat_vec = np.asarray([feat for label, feat in zip(labels, feat_vec)  if label == i])
            if _feat_vec.size > 0:
                plt.plot(_feat_vec[:,0,0,0],_feat_vec[:,0,0,1],'.',color=self.class_colors[i])
        axes = plt.gca()
        axes.axis('equal')

        return figure

    def plot_distributions(self, centers, log_covars, class_names=None, min_vals=None, max_vals=None, dims=[0,1]):

        if class_names is None:
            class_names = self.class_names

        figure = plt.figure(figsize=(8,8))
        figure.tight_layout(pad=0)

        covars = np.exp(log_covars)

        if min_vals is None:
            min_vals = (centers - 3*covars).min(axis=1)
        if max_vals is None:
            max_vals = (centers + 3*covars).max(axis=1)

        P, min_vals, max_vals = self._P_map(centers[dims,:], covars[dims,:], min_vals=min_vals[dims], max_vals=max_vals[dims])

        plt.imshow(P.max(axis=2), origin='lower', extent=(min_vals[0], max_vals[0], min_vals[1],max_vals[1]), cmap=plt.get_cmap('gray'))

        for i, class_name in zip(range(self.num_classes), class_names):
            plt.text(centers[dims[0],i], centers[dims[1],i], class_name, horizontalalignment='center', verticalalignment='center')

        axes = plt.gca()
        axes.axis('equal')

        return figure


    def plot_distributions_colored(self, centers, log_covars, class_names=None, min_vals=None, max_vals=None, dims=[0,1]):

        if class_names is None:
            class_names = self.class_names

        figure = plt.figure(figsize=(8,8))
        figure.tight_layout(pad=0)

        covars = np.exp(log_covars)

        if min_vals is None:
            min_vals = (centers - 3*covars).min(axis=1)
        if max_vals is None:
            max_vals = (centers + 3*covars).max(axis=1)

        P, min_vals, max_vals = self._P_map(centers[dims,:], covars[dims,:], min_vals=min_vals[dims], max_vals=max_vals[dims])

        P_max = P.max(axis=2)
        class_idx = np.argmax(P, axis=2)

        plt.imshow(np.expand_dims(P_max/P_max.max(), axis=-1)*np.asarray(self.class_colors)[class_idx], origin='lower', extent=(min_vals[0], max_vals[0], min_vals[1], max_vals[1]))

        for i, class_name in zip(range(self.num_classes), class_names):
            plt.text(centers[dims[0],i], centers[dims[1],i], class_name, horizontalalignment='center', verticalalignment='center')

        return figure

    def plot_classifications(self, centers, log_covars, class_names=None, min_vals=None, max_vals=None, dims=[0,1]):
      
        if class_names is None:
            class_names = self.class_names

        figure = plt.figure(figsize=(8,8))
        figure.tight_layout(pad=0)

        covars = np.exp(log_covars)

        if min_vals is None:
            min_vals = (centers - 3*covars).min(axis=1)
        if max_vals is None:
            max_vals = (centers + 3*covars).max(axis=1)

        P, min_vals, max_vals = self._P_map(centers[dims,:], covars[dims,:], min_vals=min_vals[dims], max_vals=max_vals[dims])

        class_idx = np.argmax(P, axis=2)

        plt.imshow(class_idx, origin='lower', extent=(min_vals[0], max_vals[0],min_vals[1],max_vals[1]), cmap=self.color_map)

        return figure

    def _P_map(self, centers, covars, min_vals=None, max_vals=None):

        if min_vals is None:
            min_vals = (centers - 3*covars).min(axis=1)
        if max_vals is None:
            max_vals = (centers + 3*covars).max(axis=1)

        x = np.linspace(min_vals[0], max_vals[0], 512)
        y = np.linspace(min_vals[1], max_vals[1], 512)

        X, Y = np.meshgrid(x,y)

        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        P = np.empty(X.shape + (self.num_classes,))

        for i in range(self.num_classes):
            rv = multivariate_normal(centers[:,i], np.diag(covars[:,i]))
            P[:,:,i] = rv.pdf(pos)
        
        return P, min_vals, max_vals
        

    def _classification_loss(self, D, labels_one_hot):
        with tf.name_scope('Classification_loss'):

            margin = labels_one_hot*self.alpha
            margin = margin + 1

            margin_dist = tf.math.multiply(D, margin)
            det_log_covars = tf.reduce_sum(self.log_covars, axis=0)
            margin_logits = -(0.5*det_log_covars + margin_dist)

            return tf.losses.softmax_cross_entropy(labels_one_hot, margin_logits)

    def _likelihood_regularization(self, D, labels_one_hot):
        with tf.name_scope('Likelyhood_regularization'):

            # Multiply with one-hot encoded labels to extract correct dzi's. Sum across batch
            dz = tf.reduce_sum(tf.math.multiply(D, labels_one_hot))

            # Calculate the determinant of each covariance matrix. Tricks: 1) Diagonal matrix --> prod of diagonal. 2) Stored as log of each value --> sum of diagonal
            det_log_covars = tf.reduce_sum(self.log_covars, axis=0)
            # Expand dimensions to match labels
            det_log_covars = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(det_log_covars, axis=0), axis=0), axis=0), axis=0)
            # Multiply with one-hot encoded labels to extract correct covars. Sum across batch
            det_log_covars_z = tf.reduce_sum(tf.math.multiply(det_log_covars, labels_one_hot))

            return dz + 0.5*det_log_covars_z


    def _D(self, feat_vec):
        with tf.name_scope('LGM_D'):
            X = tf.expand_dims(input=feat_vec, axis=-1)
            MU = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.centers, axis=0), axis=0), axis=0)
            COVARS = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.exp(self.log_covars), axis=0), axis=0), axis=0)

            return 0.5*tf.reduce_sum(tf.math.multiply(tf.math.divide(X - MU, COVARS), X - MU), axis=-2, keepdims=True)