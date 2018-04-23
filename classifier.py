import os
import glob
from loader import Loader
import numpy as np
import tensorflow as tf
import cv2

# Adding Seed so that random initialization is consistent
from numpy.random import seed
from tensorflow import set_random_seed


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


class ImageClassifier:

    def __init__(self):
        seed(1)
        set_random_seed(2)
        self.total_iterations = 0
        self.batch_size = 16
        self.classes = []
        self.num_classes = 0
        # 20% of the data will automatically be used for validation
        self.validation_size = 0.2
        self.image_w = 100
        self.image_h = 100
        self.num_channels = 1
        self.train_path = 'data/training1'
        self.data = None
        self.session = None
        self.accuracy = 0
        self.optimizer = None
        self.saver = None
        self.cost = 0
        self.x = None
        self.y_true = None
        self.y_true_cls = None
        self.model_dir = './model'
        self.model_name = 'MNIS'
        self.results = None

    def set_classes(self, classes):
        """ Prepare input data """
        self.classes = classes
        self.num_classes = len(classes)

    def set_image_size(self, w, h):
        self.image_w = w
        self.image_h = h

    def set_num_channels(self, count):
        self.num_channels = count

    def set_validation_size(self, size):
        self.validation_size = size

    def set_train_data_path(self, path):
        self.train_path = path

    def set_model_dir(self, path):
        self.model_dir = path

    def set_model_name(self, name):
        self.model_name = name

    def get_model_dir(self):
        if self.model_dir.startswith('/'):
            return self.model_dir
        else:
            return './{}'.format(self.model_dir)

    def get_model_name(self):
        return self.model_name

    def get_model_path(self):
        return os.path.abspath('{}/{}'.format(self.get_model_dir(), self.model_name))

    def load_data(self):
        # We shall load all the training and validation images and labels into
        # memory using openCV and use that during training
        self.data = Loader.read_train_sets(self.train_path, self.image_w, self.image_h,
                                           self.classes, self.validation_size)
        print("Complete reading input data. Will Now print a snippet of it")
        print("Number of files in Training-set:\t\t{}".format(len(self.data.train.labels)))
        print("Number of files in Validation-set:\t{}".format(len(self.data.valid.labels)))

    def create_conv_layer(self, input, num_input_channels, conv_filter_size, num_filters):
        # We shall define the weights that will be trained using create_weights function.
        weights = create_weights(
            shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
        # We create biases using the create_biases function. These are also trained.
        biases = create_biases(num_filters)

        # Creating the convolutional layer
        layer = tf.nn.conv2d(input=input,
                             filter=weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        layer += biases

        # We shall be using max-pooling.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
        # Output of pooling is fed to Relu which is the activation function for us.
        layer = tf.nn.relu(layer)
        return layer

    def create_flatten_layer(self, layer):
        # We know that the shape of the layer will be [batch_size image_w image_h num_channels]
        # But let's get it from the previous layer.
        layer_shape = layer.get_shape()

        # Number of features will be img_height * img_width* num_channels.
        # But we shall calculate it in place of hard-coding it.
        num_features = layer_shape[1:4].num_elements()

        # Now, we Flatten the layer so we shall have to reshape to num_features
        layer = tf.reshape(layer, [-1, num_features])
        return layer

    def create_fc_layer(self, input, num_inputs, num_outputs, use_relu=True):
        # Let's define trainable weights and biases.
        weights = create_weights(shape=[num_inputs, num_outputs])
        biases = create_biases(num_outputs)

        # Fully connected layer takes input x and produces wx+b.Since,
        # these are matrices, we use matmul function in Tensorflow
        layer = tf.matmul(input, weights) + biases
        if use_relu:
            layer = tf.nn.relu(layer)

        return layer

    def show_progress(self, epoch, feed_dict_train, feed_dict_validate, val_loss):
        acc = self.session.run(self.accuracy, feed_dict=feed_dict_train)
        val_acc = self.session.run(self.accuracy, feed_dict=feed_dict_validate)
        msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, " \
              "Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
        print(msg.format(epoch + 1, acc, val_acc, val_loss))

    def build_network(self):
        """
        Create a topology of network

        """
        self.session = tf.Session()
        self.x = tf.placeholder(tf.float32, shape=[None, self.image_w, self.image_h,
                                                   self.num_channels], name='x')
        # labels
        self.y_true = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y_true')
        self.y_true_cls = tf.argmax(self.y_true, 1)

        # Network graph params
        filter_size_c1 = 3
        num_filters_c1 = 32
        filter_size_c2 = 3
        num_filters_c2 = 32
        filter_size_c3 = 3
        num_filters_c3 = 64
        fc_layer_size = 128
        layer_c1 = self.create_conv_layer(self.x, self.num_channels, filter_size_c1, num_filters_c1)
        layer_c2 = self.create_conv_layer(layer_c1, num_filters_c1, filter_size_c2, num_filters_c2)
        layer_c3 = self.create_conv_layer(layer_c2, num_filters_c2, filter_size_c3, num_filters_c3)
        layer_flat = self.create_flatten_layer(layer_c3)
        elements = layer_flat.get_shape()[1:4].num_elements()
        layer_fc1 = self.create_fc_layer(layer_flat, elements, fc_layer_size, True)
        layer_fc2 = self.create_fc_layer(layer_fc1, fc_layer_size, self.num_classes, False)

        y_pred = tf.nn.softmax(layer_fc2, name='y_pred')
        y_pred_cls = tf.argmax(y_pred, 1)

        self.session.run(tf.global_variables_initializer())
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=self.y_true)
        self.cost = tf.reduce_mean(cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.cost)
        correct_prediction = tf.equal(y_pred_cls, self.y_true_cls)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.session.run(tf.global_variables_initializer())
        self.total_iterations = 0
        self.saver = tf.train.Saver()

    def train(self, num_iteration):
        """
        Train the network and create a model

        """
        self.load_data()
        self.build_network()
        for i in range(self.total_iterations, self.total_iterations + num_iteration):
            x_batch, self.y_true_batch, _, cls_batch = self.data.train.next_batch(self.batch_size)
            x_valid_batch, y_valid_batch, _, valid_cls_batch = self.data.valid.next_batch(self.batch_size)
            feed_dict_tr = {self.x: x_batch, self.y_true: self.y_true_batch}
            feed_dict_val = {self.x: x_valid_batch, self.y_true: y_valid_batch}

            self.session.run(self.optimizer, feed_dict=feed_dict_tr)

            if i % int(self.data.train.num_examples / self.batch_size) == 0:
                val_loss = self.session.run(self.cost, feed_dict=feed_dict_val)
                epoch = int(i / int(self.data.train.num_examples / self.batch_size))

                self.show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
                self.saver.save(self.session, self.get_model_path())

        self.total_iterations += num_iteration

    def load_images(self, image_dir='data/extra/', ext='*'):
        """
        Load images from directory to array

        """
        if not image_dir.endswith('/'):
            image_dir += '/'
        pattern = "{}{}".format(image_dir, ext)
        img_names = []
        images = []
        for filename in glob.iglob(pattern):
            img_names.append(filename)
            if self.num_channels == 1:
                # Gray scale
                im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            else:
                # RGB ti gray
                im = cv2.imread(filename)
                im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

            im_bw = cv2.threshold(im_gray, 185, 255, cv2.THRESH_BINARY)[1]
            image = cv2.resize(im_bw, (self.image_w, self.image_h), 0, 0, cv2.INTER_LINEAR)
            # cv2.imwrite(filename.replace(".png", "_bw.png"), image)
            image = image.reshape(self.image_w, self.image_h, self.num_channels)
            images.append(image)
        return images, img_names

    def predict(self, image_dir='data/extra/', ext='*'):
        """
        Predict images based on the computed network graph

        """
        images, img_names = self.load_images(image_dir, ext)
        images = np.array(images, dtype=np.uint8)
        images = images.astype('float32')
        images = np.multiply(images, 1.0 / 255.0)
        # The input to the network is of shape [None image_w image_h num_channels].
        x_batch = images.reshape(len(images), self.image_w, self.image_h, self.num_channels)

        # Let us restore the saved model
        self.session = tf.Session()
        # Step-1: Recreate the network graph. At this step only graph is created.
        saver = tf.train.import_meta_graph('{}/{}.meta'.format(self.get_model_dir(),
                                                               self.get_model_name()))
        # Step-2: Now let's load the weights saved using the restore method.
        saver.restore(self.session, tf.train.latest_checkpoint(self.get_model_dir()))

        # Accessing the default graph which we have restored
        graph = tf.get_default_graph()

        # Now, let's get hold of the op that we can be processed to get the output.
        # In the original network y_pred is the tensor that is the prediction of the network
        y_pred = graph.get_tensor_by_name("y_pred:0")

        # Let's feed the images to the input placeholders
        x = graph.get_tensor_by_name("x:0")
        y_true = graph.get_tensor_by_name("y_true:0")
        y_test_images = np.zeros((1, 10))

        # Creating the feed_dict that is required to be fed to calculate y_pred
        feed_dict_testing = {x: x_batch, y_true: y_test_images}
        outputs = self.session.run(y_pred, feed_dict=feed_dict_testing)

        # results
        j = 0
        self.results = []
        for result in outputs:
            item = (j, img_names[j], result)
            self.results.append(item)
            j += 1

        print('Predict..done.')

    def print_results(self):
        # print results
        if self.results:
            for item in self.results:
                print('--------------------')
                print(item[1])
                i = 0
                for p in item[2]:
                    print("{} = {:.5f}".format(i, float(p)))
                    i += 1
        else:
            print('No results.')

    def print_html_report(self):
        # generate html report page
        file = open("report_{}.html".format(self.model_name), "w")
        file.write("<html><body>")
        file.write("<h2>Results:</h2>")
        file.write("<style>td{padding:15px;border:solid 1px #000}</style>")
        file.write(self.generate_result_table())
        file.write("</body></html>")
        file.close()

    def generate_result_table(self):
        html = ""
        if self.results:
            j = 1
            for item in self.results:
                i = 0
                s = ""
                winner = 0
                number = -1
                ch = item[1].split("/")[-1][0]
                for p in item[2]:
                    s += "{} = {:.5f}<br/>".format(i, float(p))
                    if float(p) > winner:
                        winner = float(p)
                        number = i

                    i += 1
                s = "<small>{}</small>".format(s)
                path = os.path.abspath(item[1])
                cell2 = "<td>{}<br/><img src='file:///{}' /></td>".format(item[1], path)
                correct = int(ch) == number
                style = '#66FF66' if correct else '#FF6666'
                cell5 = "<td style='background-color:{}'>{}</td>".format(style, correct)
                html += "<tr><td>{}</td>{}<td>{}</td><td><h2>{}</h2></td>{}</tr>".format(j, cell2, s, number, cell5)
                j += 1

        header = "<tr><td>idx</td><td>img</td><td>stat</td><td>pred</td><td>result</td></tr>"
        return "<table style='border:1px'>{}{}</table>".format(header, html)
