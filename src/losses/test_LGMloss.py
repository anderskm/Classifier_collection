import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import src.losses.LGM

NUM_FEATS = 2
NUM_CLASSES = 5
BATCH_SIZE = 10

loss_func = src.losses.LGM.LGM(num_feats=NUM_FEATS, num_classes=NUM_CLASSES, lmbda=0.1, alpha=0.3)

feat_vec = tf.random_uniform([BATCH_SIZE, 1, 1, NUM_FEATS], minval=-1, maxval=1, dtype=tf.float32)
# Create random true labels
labels = tf.random_uniform([BATCH_SIZE, 1, 1], minval = 0, maxval = NUM_CLASSES, dtype=tf.int32)

def print_compare(label, tf_vals, np_vals):
    print('====================')
    print(label)
    print('--------------------')
    print('TF    : ' + '{:.3e}'.format(np.sum(tf_vals)))
    print('NP    : ' + '{:.3e}'.format(np.sum(np_vals)))
    print('--------------------')
    print('SAD.  : ' + '{:.3f}'.format(np.sum(np.abs(tf_vals-np_vals))))
    print('Ratio : ' + '{:.3f}'.format(np.sum(tf_vals)/np.sum(np_vals)))
    print('====================')
    print(' ')


with tf.Session() as sess:
    with tf.device('/cpu:0'):

        sess.run(tf.global_variables_initializer())

        # feat_vec = loss_func.centers

        loss = loss_func.apply(feat_vec, labels)

        tf_Lgm, feat_vec_val, labels_val, centers, log_covars, tf_D, tf_Lcls, tf_Llkd = sess.run([loss, feat_vec, labels, loss_func.centers, loss_func.log_covars, loss_func.D, loss_func.Lcls, loss_func.Llkd])

        alpha = loss_func.alpha
        lmbda = loss_func.lmbda

        print('Lgm', tf_Lgm)
        print('Lcls', tf_Lcls)
        print('Llkd', tf_Llkd)

        # np_diff = 

        # Slow but easy to understand
        np_D = np.zeros((BATCH_SIZE, 1, 1, 1, NUM_CLASSES))
        for i in range(BATCH_SIZE):
            xi = np.expand_dims(np.squeeze(feat_vec_val[i,:,:,:]), axis=-1)
            for k in range(NUM_CLASSES):
                uk = np.expand_dims(np.squeeze(centers[:,k]), axis=-1)
                Vk = np.diag(np.exp(np.squeeze(log_covars[:,k])))
                np_D[i,0,0,0,k] = 0.5*np.dot(np.dot((xi-uk).transpose(),np.linalg.inv(Vk)), (xi-uk)).squeeze()
        # Faster, but same way of calculating it as in TF
        # np_D = 0.5*np.sum((np.expand_dims(feat_vec_val, axis=-1)-centers)/np.exp(log_covars)*(np.expand_dims(feat_vec_val, axis=-1)-centers),axis=-2,keepdims=True)
        print_compare('D', tf_D, np_D)

        # Slow, but easy to understand
        np_Lkld_i = np.zeros((BATCH_SIZE,1))
        for i in range(BATCH_SIZE):
            label_i = labels_val[i,0,0].squeeze()
            det_covar = np.linalg.det(np.diag(np.exp(log_covars[:,label_i])))
            dzi = np_D[i,0,0,0,label_i]
            np_Lkld_i[i] = dzi + 0.5 * np.log(det_covar)
        np_Llkd = np_Lkld_i.sum()
        # Faster, but same way of calculating it as in TF
        # np_Llkd = np.diag(np.squeeze(np_D)[:,labels_val.squeeze()]).sum() + 0.5*np.squeeze(log_covars).sum(axis=0)[np.squeeze(labels_val)].sum()
        print_compare('Llkd', tf_Llkd, np_Llkd)

        np_Lcls_i = np.zeros((BATCH_SIZE,1))
        for i in range(BATCH_SIZE):
            label_i = labels_val[i,0,0].squeeze()
            Di = np_D[i,0,0,0,:]
            Di[label_i] *= 1+alpha

            e = -(0.5*np.sum(log_covars, axis=0) + Di)
            e = e-e.max() # See trick here (stable Softmax function): https://deepnotes.io/softmax-crossentropy

            # See note here for calculating softmax cross entropy:
            # https://pytorch.org/docs/stable/nn.html?highlight=crossentropy#torch.nn.CrossEntropyLoss
            np_Lcls_i[i] = -e[label_i] + np.log(np.sum(np.exp(e)))

        np_Lcls = np_Lcls_i.sum()/BATCH_SIZE
        print_compare('Lcls', tf_Lcls, np_Lcls)

        print_compare('Lgm', tf_Lgm, (np_Lcls + lmbda*np_Llkd)/BATCH_SIZE)

        figure = loss_func.plot_feat_vec(feat_vec_val, labels_val)
        img = loss_func.figure_to_array(figure)

        figure = loss_func.plot_distributions(centers, log_covars)

        figure = loss_func.plot_classifications(centers, log_covars)

        figure = loss_func.plot_distributions_colored(centers, log_covars)

plt.show()

print('done...')


# print(loss)