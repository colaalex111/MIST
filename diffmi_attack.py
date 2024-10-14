import numpy as np
import tensorflow as tf
import tensorflow.compat.v1.keras.layers as keraslayers
from tensorflow.compat.v1.train import Saver
import tqdm
from model_utils import convert_model_from_pytorch_to_tensorflow, sobel, mmd_loss
from model_utils import *
from scipy.special import softmax


def diffmi_attack(user_list, target_model, batch_size, output_file):
    ### notice that this attack is doing attack per batch (20 samples in one batch as default, maybe we need to change to 200)
    keras_target_model = convert_model_from_pytorch_to_tensorflow(target_model)

    # print ("model converted")

    # keras_target_model.summary()
    # input_shape = (32, 32, 3)

    all_m_true = []
    all_m_pred = []

    for this_user in user_list:
        min_len = min(len(this_user.train_data), len(this_user.test_data))

        x_ = np.r_[this_user.train_data[:min_len], this_user.test_data[:min_len]]
        # maybe we need to reshape the data???

        y_true = np.r_[this_user.train_label[:min_len], this_user.test_label[:min_len]]
        ### reshape y_true to be a one-hot vector for the diff_Mem_attack call
        n_values = np.max(y_true) + 1  ### this should be 10 or 100
        # print ("n_values for diffmi attack {v} ".format(v=n_values))
        converted_y_true = np.eye(n_values)[y_true]
        ### check conversion -- conversion correct
        # print (y_true[:10])
        # print (converted_y_true[:10])

        m_true = np.r_[np.ones(min_len), np.zeros(min_len)]

        m_true, m_pred, mix, nonMem = diff_Mem_attack(x_, converted_y_true, m_true, keras_target_model,
                                                      batch_size=batch_size)

        all_m_true.append(m_true)
        all_m_pred.append(m_pred)

    all_m_true = np.array(all_m_true).flatten()
    all_m_pred = np.array(all_m_pred).flatten()

    from sklearn.metrics import accuracy_score
    # print (f"blindMI acc {accuracy_score(all_m_true,all_m_pred)*100}")
    output_file.write(f"blindMI acc {accuracy_score(all_m_true, all_m_pred) * 100}\n")

    return all_m_true, all_m_pred


def diff_Mem_attack(x_, y_true, m_true, target_model, non_Mem_Generator=sobel, batch_size=20):
    '''
    Attck the target with BLINDMI-DIFF-W, BLINDMI-DIFF with gernerated non-member.
    The non-member is generated by randomly chosen data and the number is 20 by default.
    If the data has been shuffled, please directly remove the process of shuffling.
    :param target_model: the model that will be attacked
    :param x_: the data that target model may used for training
    :param y_true: the label of x_
    :param m_true: one of 0 and 1, which represents each of x_ has been trained or not.
    :param non_Mem_Generator: the method to generate the non-member data. The default non-member generator
    is Sobel.
    :return:  Tensor arrays of results
    '''

    # print (x_.shape)
    # print (y_true.shape)
    # print (m_true.shape)

    ### need to check the np.unique(y_true) for members and non-members

    # print ("doing diff mem attack")

    y_pred = softmax(target_model.predict(x_), axis=1)
    # print (np.sum(y_pred[0]))
    # print (y_pred[0])
    # print (target_model.predict(x_)[0])
    mix = np.c_[y_pred[y_true.astype(bool)], np.sort(y_pred, axis=1)[:, ::-1][:, :2]]

    nonMem_index = np.random.randint(0, x_.shape[0], size=batch_size)

    nonMem_pred = target_model.predict(non_Mem_Generator(x_[nonMem_index]))
    # print (nonMem_pred)
    nonMem_pred = softmax(nonMem_pred, axis=1)
    # print (nonMem_pred)
    ### sobel here is an edge detection algo

    nonMem = tf.convert_to_tensor(np.c_[nonMem_pred[y_true[nonMem_index].astype(bool)],
                                        np.sort(nonMem_pred, axis=1)[:, ::-1][:, :2]])
    ### prob of the label + top 2 prob ??? need to verify for nonMem

    data = tf.data.Dataset.from_tensor_slices((mix, m_true)).shuffle(buffer_size=x_.shape[0]). \
        batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    ### each element in mix and each element in m_true is combined as one tensor
    ### each batch is formed as 20 elements as default

    m_pred, m_true = [], []
    mix_shuffled = []
    for (mix_batch, m_true_batch) in data:
        m_pred_batch = np.ones(mix_batch.shape[0])
        m_pred_epoch = np.ones(mix_batch.shape[0])
        nonMemInMix = True

        # print (tf.shape(mix_batch),tf.shape(m_true_batch))

        while nonMemInMix:
            mix_epoch_new = mix_batch[m_pred_epoch.astype(bool)]  # take all members in this new batch
            dis_ori = mmd_loss(nonMem, mix_epoch_new, weight=1)
            nonMemInMix = False
            for index, item in enumerate(mix_batch):
                if m_pred_batch[index] == 1:
                    nonMem_batch_new = tf.concat([nonMem, [mix_batch[index]]], axis=0)
                    ## new batch of non-mem set (nonMem + 1 testing instance)
                    mix_batch_new = tf.concat([mix_batch[:index], mix_batch[index + 1:]], axis=0)
                    ## new batch of mem set (this batch without the testing instance)
                    m_pred_without = np.r_[m_pred_batch[:index], m_pred_batch[index + 1:]]
                    ## why do this?
                    mix_batch_new = mix_batch_new[m_pred_without.astype(bool, copy=True)]
                    ## new batch of mem set (this batch w/o the testing instance and only members are included)
                    dis_new = mmd_loss(nonMem_batch_new, mix_batch_new, weight=1)
                    ## new distance
                    if dis_new > dis_ori:
                        nonMemInMix = True
                        m_pred_epoch[index] = 0
            m_pred_batch = m_pred_epoch.copy()

        ### finishing process this small batch

        mix_shuffled.append(mix_batch)
        m_pred.append(m_pred_batch)
        m_true.append(m_true_batch)
    return np.concatenate(m_true, axis=0), np.concatenate(m_pred, axis=0), \
           np.concatenate(mix_shuffled, axis=0), nonMem
