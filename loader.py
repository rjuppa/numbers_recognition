import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np


class DataSet(object):
    def __init__(self, images, labels, img_names, cls):
        self._num_examples = images.shape[0]

        self._images = images
        self._labels = labels
        self._img_names = img_names
        self._cls = cls
        self._epochs_done = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def img_names(self):
        return self._img_names

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        """ Return the next `batch_size` examples from this data set. """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # After each epoch we update this
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end], \
               self._img_names[start:end], self._cls[start:end]


class Loader(object):

    @staticmethod
    def load_train(train_path, image_w, image_h, classes):
        images = []
        labels = []
        img_names = []
        cls = []

        print('Going to read training images')
        for fields in classes:
            index = classes.index(fields)
            print('Now going to read {} files (Index: {})'.format(fields, index))
            path = os.path.join(train_path, fields, '*g')
            files = glob.glob(path)
            for fl in files:
                try:
                    if fl.find("_bw.p", 0) > 0:
                        os.remove(fl)
                        continue
                except:
                    pass
                im_gray = cv2.imread(fl, cv2.IMREAD_GRAYSCALE)
                im_bw = cv2.threshold(im_gray, 185, 255, cv2.THRESH_BINARY)[1]
                image = cv2.resize(im_bw, (image_w, image_h), 0, 0, cv2.INTER_LINEAR)
                # cv2.imwrite(fl.replace(".png", "_bw.png"), image)
                image = image.reshape(image_w, image_h, 1)
                image = image.astype(np.float32)
                image = np.multiply(image, 1.0 / 255.0)
                images.append(image)
                label = np.zeros(len(classes))
                label[index] = 1.0
                labels.append(label)
                flbase = os.path.basename(fl)
                img_names.append(flbase)
                cls.append(fields)
        images = np.array(images)
        labels = np.array(labels)
        img_names = np.array(img_names)
        cls = np.array(cls)

        return images, labels, img_names, cls

    @staticmethod
    def read_train_sets(train_path, image_w, image_h, classes, val_size):
        class DataSets(object):
            pass

        images, labels, img_names, cls = Loader.load_train(train_path, image_w, image_h, classes)
        images, labels, img_names, cls = shuffle(images, labels, img_names, cls)

        if isinstance(val_size, float):
            val_size = int(val_size * images.shape[0])

        val_images = images[:val_size]
        val_labels = labels[:val_size]
        val_img_names = img_names[:val_size]
        val_cls = cls[:val_size]

        train_images = images[val_size:]
        train_labels = labels[val_size:]
        train_img_names = img_names[val_size:]
        train_cls = cls[val_size:]

        data_sets = DataSets()
        data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
        data_sets.valid = DataSet(val_images, val_labels, val_img_names, val_cls)
        return data_sets
