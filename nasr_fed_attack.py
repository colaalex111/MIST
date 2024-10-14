'''
reference: https://github.com/privacytrustlab/ml_privacy_meter
'''
import numpy as np
import os
import copy
import ml_privacy_meter
import tensorflow as tf
import tensorflow.compat.v1.keras.layers as keraslayers
from tensorflow.compat.v1.train import Saver
from model_utils import convert_model_from_pytorch_to_tensorflow
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import metrics
from scipy.special import softmax
from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score


def load_CIFAR(model_mode='TargetModel'):
    """
    Loads CIFAR-100 or CIFAR-10 dataset and maps it to Target Model and Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :param num_classes: one of 10 and 100 and the default value is 100
    :return: Tuple of numpy arrays:'(x_train, y_train), (x_test, y_test), member'.
    :raise: ValueError: in case of invalid `model_mode`.
    """
    if model_mode not in ['TargetModel', 'ShadowModel']:
        raise ValueError('model_mode must be one of TargetModel, ShadowModel.')

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    if model_mode == "TargetModel":
        (x_train, y_train), (x_test, y_test) = (x_train[40000:50000], y_train[40000:50000]), \
                                               (x_test, y_test)
    elif model_mode == "ShadowModel":
        (x_train, y_train), (x_test, y_test) = (x_train[:10000], y_train[:10000]), \
                                               (x_train[10000:20000], y_train[10000:20000])

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=100)
    m_train = np.ones(y_train.shape[0])

    y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)
    m_test = np.zeros(y_test.shape[0])

    member = np.r_[m_train, m_test]
    return x_train, y_train, x_test, y_test, member


def test_converted_model(model, user_list):
    print ("test converted model:")

    def normalize(x_test, means, stddevs):
        """
        Normalizes data using means and stddevs
        """
        # x_test = (x_test / 255 - means) / stddevs
        # print (x_test.type)
        x_test = np.array(x_test).astype(np.float32)
        # x_test = np.transpose(x_test,[0,3,1,2])
        # print (type(x_test),x_test.dtype,x_test.shape)
        x_test = x_test / 255
        # print (np.amax(x_test),np.amin(x_test))
        expanded_means = np.zeros_like(x_test)
        for idx, this_mean in enumerate(means):
            expanded_means[:, :, :, idx] = this_mean
        x_test = x_test - expanded_means
        # print (np.amax(x_test),np.amin(x_test))
        expanded_stddevs = np.zeros_like(x_test)
        for idx, this_stddev in enumerate(stddevs):
            expanded_stddevs[:, :, :, idx] = this_stddev
        x_test = x_test / expanded_stddevs
        # print (np.amax(x_test),np.amin(x_test))
        return x_test

    def evaluate(x_test, y_test):

        model.compile(loss='categorical_crossentropy',
                      metrics=[metrics.CategoricalAccuracy(), metrics.Precision(), metrics.Recall()])

        # print (model.summary())

        # print ("pre normalized data:",np.argmax(x_test),np.argmin(x_test))
        normalized_x = normalize(x_test=x_test, means=(0.4914, 0.4822, 0.4465), stddevs=(0.2023, 0.1994, 0.2010))
        # print ("post normalized data:",np.argmax(normalized_x),np.argmin(normalized_x))

        # print (normalized_x.shape)

        y_pred = model.predict(normalized_x)

        # print (y_test.shape)
        # print (y_pred.shape)

        if (len(y_test.shape) > 1):
            y_test = np.argmax(y_test, axis=1)

        y_pred = softmax(y_pred, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

        print (y_test.shape)
        print (y_pred.shape)
        print (accuracy_score(y_test, y_pred) * 100)
        print (classification_report(y_test, y_pred))
        # loss, accuracy, precision, recall = model.evaluate(x_test, y_test, verbose=1)
        # F1_Score = 2 * (precision * recall) / (precision + recall)
        # print('loss:%.4f accuracy:%.4f precision:%.4f recall:%.4f F1_Score:%.4f'
        #      % (loss, accuracy, precision, recall, F1_Score))

    x_test, y_test = user_list[0].test_data, user_list[0].test_label
    # print ("evaluate test data:")
    evaluate(x_test, y_test)

    # _,_,x_test,y_test,_ = load_CIFAR(model_mode='TargetModel')
    # print ("evaluate test data:")
    # evaluate(x_test, y_test)
    ### model is fine !!!!!


def nasr_fed_attack(user_list, target_model, dataset_name, output_file, model_name='alexnet', mid_str=''):
    ### since this function is actually tensorflow based,
    ### we need to convert the target model from pytorch to tensorflow to make it work
    keras_target_model = convert_model_from_pytorch_to_tensorflow(target_model)

    # test_converted_model(keras_target_model,user_list)

    # for idx,this_layer in enumerate(keras_target_model.layers):
    #    print (idx,this_layer,this_layer.input_shape,this_layer.output_shape)

    # keras_target_model.summary()

    input_shape = (32, 32, 3)
    ### now we need to build an attack model
    ### here we assume that the attacker uses:
    #  the final layer gradients,
    #  the final layer outputs,
    #  the loss values,
    #  the label values

    # `saved_path` is required for obtaining the training data that was used to
    # train the target classification model. This is because
    # the datapoints that form the memberset of the training data of the attack
    # model has to be a subset of the training data of target classification model.
    # User can store the training data wherever he/she wants but the only requirement
    # is that the file has to be stored in '.npy' format. The contents should be of
    # the same format as the .txt file of the dataset.
    # saved_path = "/home/lijiacheng/ml_privacy_meter-master/datasets/cifar100_train.txt.npy"

    # Similar to `saved_path` being used to form the memberset for attack model,
    # `dataset_path` is used for forming the nonmemberset of the training data of
    # attack model.
    # dataset_path = '/home/lijiacheng/ml_privacy_meter-master/datasets/cifar100.txt'

    member_corr_count = 0
    nonmember_corr_count = 0

    for this_user in user_list:

        prefix = "/home/lijiacheng/fed-exp-data/" + mid_str

        ### save the test data file

        testing_data_saved_path = prefix + dataset_name + ".txt"

        # create the testdict
        testdict = {}
        copied_data = copy.deepcopy(this_user.test_data)
        testdict['data'] = copied_data
        testdict['fine_labels'] = this_user.test_label

        # print (copied_data.shape)

        with open(testing_data_saved_path, "w") as f:
            for i in range(len(testdict['data'])):
                c0 = testdict['data'][i][:, :, 0].flatten()
                # print (c0.shape)
                c1 = testdict['data'][i][:, :, 1].flatten()
                c2 = testdict['data'][i][:, :, 2].flatten()

                a = ','.join([str(c) for c in c0]) + ';' + \
                    ','.join([str(c) for c in c1]) + ';' + \
                    ','.join([str(c) for c in c2]) + ';' + \
                    str(testdict['fine_labels'][i])
                f.write(a + "\n")

        ### save the train data file
        tmp_name = prefix + dataset_name + ".txt.tmp"
        training_data_saved_path = prefix + dataset_name + ".npy"

        # create the traindict
        traindict = {}
        copied_data = copy.deepcopy(this_user.train_data)
        traindict['data'] = copied_data
        traindict['fine_labels'] = this_user.train_label

        with open(tmp_name, "w") as f:
            for i in range(len(traindict['data'])):
                c0 = traindict['data'][i][:, :, 0].flatten()
                c1 = traindict['data'][i][:, :, 1].flatten()
                c2 = traindict['data'][i][:, :, 2].flatten()

                a = ','.join([str(c) for c in c0]) + ';' + \
                    ','.join([str(c) for c in c1]) + ';' + \
                    ','.join([str(c) for c in c2]) + ';' + \
                    str(traindict['fine_labels'][i])
                f.write(a + "\n")

        def extract(filepath):
            """
            """
            with open(filepath, "r") as f:
                dataset = f.readlines()
            dataset = map(lambda i: i.strip('\n').split(';'), dataset)
            dataset = np.array(list(dataset))

            return dataset

        dataset = extract(tmp_name)
        os.remove(tmp_name)
        np.save(training_data_saved_path, dataset)
        
        # print(dataset.shape,dataset[0].shape,dataset[0][0],dataset[0][-1])
        # print (this_user.test_data[0])
        # print (this_user.test_label[:10])

        # print ("START NASR ATTACK")

        datahandlerA = ml_privacy_meter.utils.attack_data.attack_data(dataset_path=testing_data_saved_path,
                                                                      member_dataset_path=training_data_saved_path,
                                                                      batch_size=2,
                                                                      attack_percentage=10, input_shape=input_shape,
                                                                      normalization=True)

        datahandlerA.means, datahandlerA.stddevs = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]

        layers_to_exploit = [22] if (model_name == 'alexnet') else [-1]
        gradients_to_exploit = [6] if (model_name == 'alexnet') else [-1]

        attackobj = ml_privacy_meter.attack.meminf.initialize(
            target_train_model=keras_target_model,
            target_attack_model=keras_target_model,
            train_datahandler=datahandlerA,
            attack_datahandler=datahandlerA,
            # optimizer="adam",
            layers_to_exploit=layers_to_exploit,
            gradients_to_exploit=gradients_to_exploit,
            exploit_loss=True,
            exploit_label=True,
            # learning_rate=0.001,
            epochs=5
        )

        attackobj.train_attack()
        mpreds, nmpreds = attackobj.test_attack()

        mpreds = np.array(mpreds).flatten()
        nmpreds = np.array(nmpreds).flatten()

        # print (mpreds.shape,nmpreds.shape)
        # print (mpreds)

        member_corr_count += len(np.arange(len(mpreds))[mpreds > 0.5])
        nonmember_corr_count += len(np.arange(len(nmpreds))[nmpreds <= 0.5])

    acc = (member_corr_count + nonmember_corr_count) / (len(mpreds) + len(nmpreds)) * (100 / len(user_list))
    print (f"nasr attack acc {acc}")
    output_file.write(f"nasr attack acc {acc} \n")

    # return mpreds,nmpreds
