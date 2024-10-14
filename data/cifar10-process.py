import pickle
import numpy as np
import os

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data(filename):
    data = unpickle(filename)

    raw_images = data[b'data']
    classes = data[b'labels']
    raw_images = np.array(raw_images).astype('float32')
    raw_images = np.reshape(raw_images,(-1,3,32,32))
    raw_images = raw_images.transpose([0,2,3,1])
    return raw_images,np.array(classes)



if __name__ == '__main__':
   
    images = np.zeros(shape=[50000, 32, 32, 3], dtype=np.float32)
    cls = np.zeros(shape=[50000], dtype=np.int32)

    # Begin-index for the current batch.
    begin = 0

    # For each data-file.
    for i in range(5):
        # Load the images and class-numbers from the data-file.
        images_batch, cls_batch = load_data(filename="data_batch_" + str(i + 1))

        # Number of images in this batch.
        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        # Store the images into the array.
        images[begin:end, :] = images_batch

        # Store the class-numbers into the array.
        cls[begin:end] = cls_batch

        # The begin-index for the next batch is the current end-index.
        begin = end

    print (images.shape)
    print (cls.shape)

    np.save('cifar10_train_data.npy',images)
    np.save('cifar10_train_label.npy',cls)

    images, cls = load_data(filename="test_batch")

    print (images.shape)
    print (cls.shape)

    np.save('cifar10_test_data.npy',images)
    np.save('cifar10_test_label.npy',cls)

    print ("completed!")
