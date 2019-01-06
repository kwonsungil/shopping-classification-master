import tensorflow as tf
import numpy as np
import os
import datetime
import global_constants as gcon

class kakao(object):
    def __init__(self, vocap_size, training=True):
        # Placeholders for input, output and dropout
        self.sequence_length = 40
        self.embedding_size = 32
        self.feature_size = 2048
        self.initial_learning_rate = 0.1
        self.decay_steps = 2000
        self.decay_rate = 0.9
        self.staircase = False
        self.logs_dir = gcon.base_model_dir
        os.makedirs(self.logs_dir, exist_ok=True)
        self.filters = 512
        # self.dropout_keep_prob = 1
        self.alpha = 0.1
        self.num_classes = 4215
        self.bcateid_size = 57
        self.mcateid_size = 552
        self.scateid_size = 3190
        self.dcateid_size = 404
        self.hidden_size = 200
        self.summary_step = 1000
        self.vocap_size = vocap_size
        self.avg_b_loss = .0
        self.avg_m_loss = .0
        self.avg_s_loss = .0
        self.avg_d_loss = .0
        self.avg_c_loss = .0
        self.avg_l_loss = .0
        self.avg_b_acc = .0
        self.avg_m_acc = .0
        self.avg_s_acc = .0
        self.avg_d_acc = .0
        self.avg_c_acc = .0
        self.avg_l_acc = .0
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_feature = tf.placeholder(tf.float32, [None, self.feature_size], name="input_feature")
        self.embeddings = tf.concat((tf.zeros([1, self.embedding_size]),
                                        tf.Variable(tf.random_uniform([self.vocap_size-1, self.embedding_size], -1.0, 1.0),
                                                    trainable=True, name='embeddings')), axis=0)

        self.embeddings_batch =tf.nn.embedding_lookup(self.embeddings, self.input_x)

        self.input_feature = tf.placeholder(tf.float32, [None, self.feature_size], name="input_feature")
        self.training = training
        with tf.variable_scope('image'):
            self.feature_fc_bn = self.build_image_model()
        with tf.variable_scope('model_1'):
            self.logits_1, self.logits_2, self.logits_3, self.logits_4 = self.build_model_1()
        with tf.variable_scope('model_2'):
            self.logits_5 = self.build_model_2()
        with tf.variable_scope('last_layer'):
            self.logits_6 = self.build_last_layer()

        gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     print(self.embeddings.eval())
        #     print(self.embeddings.eval()[0])

        self.saver = tf.train.Saver(max_to_keep=40)
        if self.training:
            self.bcateid = tf.placeholder(tf.int32, [None], name="bcateid")
            self.mcateid = tf.placeholder(tf.int32, [None], name="mcateid")
            self.scateid = tf.placeholder(tf.int32, [None], name="scateid")
            self.dcateid = tf.placeholder(tf.int32, [None], name="dcateid")
            self.label = tf.placeholder(tf.int32, [None], name="label")
            self.learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.writer = tf.summary.FileWriter(self.logs_dir)
            self.writer.add_graph(self.sess.graph)
            # self.learning_rate = tf.train.exponential_decay(
            #     self.initial_learning_rate, self.global_step, self.decay_steps,
            #     self.decay_rate, self.staircase, name='learning_rate')
            with tf.variable_scope('model_1'):
                self.model_1_loss = self.compute_model_1_loss()
            with tf.variable_scope('model_2'):
                self.class_loss = self.compute_model_2_loss()
            with tf.variable_scope('last_layer'):
                self.last_class_loss = self.compute_last_layer_loss()

            # self.total_loss = self.model_1_loss + self.class_loss
            self.total_loss = self.model_1_loss + self.class_loss + self.last_class_loss

            # self.model_1_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.model_1_loss, name='model_1_optimizer')
            # self.model_2_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.class_loss, global_step=self.global_step, name='model_2_optimizer')
            # self.model_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss, global_step=self.global_step, name='model_optimizer')
            self.last_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss, global_step=self.global_step,  name='last_optimizer')

            # tf.summary.scalar('total_loss', self.model_1_loss)
            tf.summary.scalar('total_loss', self.class_loss)
            # tf.summary.scalar('total_loss', self.last_class_loss)
            tf.summary.scalar('learning_rate', self.learning_rate)
            self.summary_op = tf.summary.merge_all()

        filename = tf.train.latest_checkpoint(self.logs_dir)
        # filename = os.path.join('F:\\Dataset\\kakao\\shopping-classification-master\\logs', 'model.ckpt-1401')
        self.sess.run(tf.global_variables_initializer())
        if filename is not None:
            print('restore from : ', filename)
            self.saver.restore(self.sess, filename)


    def lstm_layer(self, name, inputs, output_size, initial_state=None, trainable=True):
        with tf.variable_scope(name):
            cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)

            # if initial_state is None:
            #     initial_state = cell.zero_state(None, tf.float32)

            # cell = tf.nn.rnn_cell.DropoutWrapper(cell, self.dropout_keep_prob)
            outputs, states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, dtype=tf.float32)

            lstm_weight = tf.get_variable('lstm_weight', shape=[int(inputs.shape[1]) * self.hidden_size, output_size],
                                          initializer=tf.truncated_normal_initializer(),
                                          trainable=trainable)
            lstm_bias = tf.get_variable('lstm_bias', shape=output_size,
                                        initializer=tf.truncated_normal_initializer(), trainable=trainable)
            return tf.add(tf.matmul(tf.reshape(outputs, [-1, int(inputs.shape[1]) * self.hidden_size]), lstm_weight),
                          lstm_bias), states


    def conv_layer(self, name, inputs, filters, size_1, size_2, stride=1, in_channel=1, trainable=True):
        with tf.variable_scope(name):
            weight = tf.get_variable('conv_weight', shape=[size_1, size_2, in_channel, filters],
                                     initializer=tf.truncated_normal_initializer(),
                                     trainable=trainable)
            bias = tf.get_variable('conv_bias', shape=filters,
                                   initializer=tf.truncated_normal_initializer(),
                                   trainable=trainable)
            # weight = tf.Variable(tf.truncated_normal([size, self.embedding_size, in_channel, filters], stddev=0.1), trainable=trainable)
            # biases = tf.Variable(tf.constant(0.1, shape=[filters]), trainable=trainable)
            conv = tf.nn.conv2d(inputs, weight, strides=[1, stride, stride, 1], padding='VALID',
                                name='conv')
            conv_biased = tf.add(conv, bias, name='conv_biased')
            # return tf.nn.relu(conv_biased)
            return conv_biased

    def batch_norm_conv(self, name, inputs, trainable=True):
        with tf.variable_scope(name):
            variance_epsilon = tf.constant(0.0003, name="epsilon")
            beta = tf.Variable(tf.constant(0.0, shape=[int(inputs.shape[3])]), trainable=trainable, dtype=tf.float32, name="beta")
            gamma = tf.Variable(tf.constant(1.0, shape=[int(inputs.shape[3])]), trainable=trainable, dtype=tf.float32, name="gamma")
            moving_mean, moving_variance = tf.nn.moments(inputs, axes=[0, 1, 2])
            # moving_mean = tf.Variable(moving_mean, trainable=trainable, dtype=tf.float32, name="moving_mean")
            # moving_variance = tf.Variable(moving_variance, trainable=trainable, dtype=tf.float32, name="moving_variance")
            bn = tf.nn.batch_normalization(inputs, moving_mean, moving_variance, beta, gamma,
                                           variance_epsilon, name='BatchNorm')

            # tf.summary.histogram('moving_mean', moving_mean)
            # tf.summary.histogram('moving_variance', moving_variance)
            # tf.summary.histogram('beta', beta)
            # tf.summary.histogram('gamma', gamma)
            return tf.nn.relu(bn)

    def batch_norm_fc(self, name, inputs, activate=None, trainable=True):
        with tf.variable_scope(name):
            variance_epsilon = tf.constant(0.0001, name="epsilon")
            beta = tf.Variable(tf.constant(0.0, shape=[int(inputs.shape[1])]), trainable=trainable, dtype=tf.float32, name="beta")
            gamma = tf.Variable(tf.constant(1.0, shape=[int(inputs.shape[1])]), trainable=trainable, dtype=tf.float32, name="gamma")
            moving_mean, moving_variance = tf.nn.moments(inputs, axes=[0])
            # moving_mean = tf.Variable(moving_mean, trainable=trainable, dtype=tf.float32, name="moving_mean")
            # moving_variance = tf.Variable(moving_variance, trainable=trainable, dtype=tf.float32, name="moving_variance")
            bn = tf.nn.batch_normalization(inputs, moving_mean, moving_variance, beta, gamma,
                                           variance_epsilon, name='BatchNorm')

            tf.summary.histogram('moving_mean', moving_mean)
            tf.summary.histogram('moving_variance', moving_variance)
            tf.summary.histogram('beta', beta)
            tf.summary.histogram('gamma', gamma)
            return bn if activate is None else tf.nn.relu(bn)

    def max_pooling(self, name, inputs):
        with tf.variable_scope(name):
            return tf.nn.max_pool(
                inputs,
                ksize=[1, int(inputs.shape[1]), 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")

    def fc_layer(self, name, inputs, outputs, activate=None, trainable=True):
        # weight = tf.Variable(tf.truncated_normal([dim, hiddens], stddev=0.1), trainable=trainable)
        with tf.variable_scope(name):
            weight = tf.get_variable('fc_weight', shape=[int(inputs.shape[1]), outputs],
                                     initializer=tf.truncated_normal_initializer(),
                                     trainable=trainable)
            bias = tf.get_variable('fc_bias', shape=outputs,
                                   initializer=tf.truncated_normal_initializer(),
                                   trainable=trainable)
            return tf.add(tf.matmul(inputs, weight), bias) if activate is None else tf.nn.relu(
                tf.add(tf.matmul(inputs, weight), bias))

    def dropout(self, name, inputs):
        with tf.variable_scope(name):
            return tf.nn.dropout(inputs, keep_prob=self.dropout_keep_prob)

    def build_image_model(self):
        feature_fc = self.fc_layer('feature_fc', self.input_feature, self.filters, activate=None)
        feature_fc_bn = self.batch_norm_fc('feature_fc_bn', feature_fc, activate='relu')
        return feature_fc_bn

    def build_model_2(self):

        dot_product = tf.divide(tf.matmul(self.embeddings_batch, self.embeddings_batch, transpose_a=False, transpose_b= True), tf.sqrt(tf.cast(self.embedding_size, dtype=tf.float32)))
        dot_product = tf.reduce_sum(dot_product, axis=2, keep_dims=True)
        dot_product = tf.nn.softmax(dot_product)
        dot_product = tf.multiply(dot_product, self.embeddings_batch)

        product_conv_2 = self.conv_layer('product_conv_2', tf.expand_dims(dot_product, -1), self.filters, size_1 = 2, size_2=self.embedding_size)
        product_conv_3 = self.conv_layer('product_conv_3', tf.expand_dims(dot_product, -1), self.filters, size_1 = 3, size_2=self.embedding_size)
        product_conv_4 = self.conv_layer('product_conv_4', tf.expand_dims(dot_product, -1), self.filters, size_1 = 4, size_2=self.embedding_size)
        product_conv_5 = self.conv_layer('product_conv_5', tf.expand_dims(dot_product, -1), self.filters, size_1 = 5, size_2=self.embedding_size)

        product_pooling_2 = self.max_pooling('product_pooling_2', product_conv_2)
        product_pooling_3 = self.max_pooling('product_pooling_3', product_conv_3)
        product_pooling_4 = self.max_pooling('product_pooling_4', product_conv_4)
        product_pooling_5 = self.max_pooling('product_pooling_5', product_conv_5)

        product_pooling_2 = tf.reshape(product_pooling_2, shape=[-1, self.filters])
        product_pooling_3 = tf.reshape(product_pooling_3, shape=[-1, self.filters])
        product_pooling_4 = tf.reshape(product_pooling_4, shape=[-1, self.filters])
        product_pooling_5 = tf.reshape(product_pooling_5, shape=[-1, self.filters])


        # feature_fc = self.fc_layer('feature_fc', self.input_feature, self.filters, activate=None)
        # feature_fc_bn = self.batch_norm_fc('feature_fc_bn', feature_fc, activate='relu')

        flat = tf.concat([product_pooling_2, product_pooling_3, product_pooling_4, product_pooling_5, self.feature_fc_bn], 1)
        fc_1 = self.fc_layer('fc_1', flat, 5000, activate=None)
        fc_bn_1 = self.batch_norm_fc('fc_bn_1', fc_1, activate='relu')

        fc_2 = self.fc_layer('fc_2', fc_bn_1, 4500, activate=None)
        fc_bn_2 = self.batch_norm_fc('fc_bn_2', fc_2, activate='relu')

        fc_3 = self.fc_layer('fc_3', fc_bn_2, self.num_classes, activate=None)
        fc_bn_3 = self.batch_norm_fc('fc_bn_3', fc_3, activate=None)

        return fc_bn_3

    def build_model_1(self):
        print('self.embeddings_batch : ', self.embeddings_batch)

        dot_product = tf.divide(tf.matmul(self.embeddings_batch, self.embeddings_batch, transpose_a=False, transpose_b= True), tf.sqrt(tf.cast(self.embedding_size, dtype=tf.float32)))
        dot_product = tf.reduce_sum(dot_product, axis=2, keep_dims=True)
        dot_product = tf.nn.softmax(dot_product)
        dot_product = tf.multiply(dot_product, self.embeddings_batch)

        print(dot_product)
        product_conv_2 = self.conv_layer('product_conv_2', tf.expand_dims(dot_product, -1), self.filters, size_1 = 2, size_2=self.embedding_size)
        product_conv_3 = self.conv_layer('product_conv_3', tf.expand_dims(dot_product, -1), self.filters, size_1 = 3, size_2=self.embedding_size)
        product_conv_4 = self.conv_layer('product_conv_4', tf.expand_dims(dot_product, -1), self.filters, size_1 = 4, size_2=self.embedding_size)
        product_conv_5 = self.conv_layer('product_conv_5', tf.expand_dims(dot_product, -1), self.filters, size_1 = 5, size_2=self.embedding_size)

        product_pooling_2 = self.max_pooling('product_pooling_2', product_conv_2)
        product_pooling_3 = self.max_pooling('product_pooling_3', product_conv_3)
        product_pooling_4 = self.max_pooling('product_pooling_4', product_conv_4)
        product_pooling_5 = self.max_pooling('product_pooling_5', product_conv_5)
        print('product_pooling_2 : ', product_pooling_2)

        product_pooling_2 = tf.reshape(product_pooling_2, shape=[-1, self.filters])
        product_pooling_3 = tf.reshape(product_pooling_3, shape=[-1, self.filters])
        product_pooling_4 = tf.reshape(product_pooling_4, shape=[-1, self.filters])
        product_pooling_5 = tf.reshape(product_pooling_5, shape=[-1, self.filters])

        # feature_fc = self.fc_layer('feature_fc', self.input_feature, self.filters, activate=None)
        # feature_fc_bn = self.batch_norm_fc('feature_fc_bn', feature_fc, activate='relu')
        flat = tf.concat([product_pooling_2, product_pooling_3, product_pooling_4, product_pooling_5, self.feature_fc_bn], 1)

        fc_1 = self.fc_layer('fc_1', flat, 4000, activate=None)
        fc_bn_1 = self.batch_norm_fc('fc_bn_1', fc_1, activate='relu')

        fc_b_1 = self.fc_layer('fc_b_1', fc_bn_1, 500, activate=None)
        fc_b_bn_1 = self.batch_norm_fc('fc_b_bn_1', fc_b_1, activate='relu')
        fc_b_2 = self.fc_layer('fc_b_2', fc_b_bn_1, self.bcateid_size, activate=None)
        fc_b_bn_2 = self.batch_norm_fc('fc_b_bn_2', fc_b_2, activate=None)

        fc_m_1 = self.fc_layer('fc_m_1', fc_bn_1, 1000, activate=None)
        fc_m_bn_1 = self.batch_norm_fc('fc_m_bn_1', fc_m_1, activate='relu')
        fc_m_2 = self.fc_layer('fc_m_2', fc_m_bn_1, self.mcateid_size, activate=None)
        fc_m_bn_2 = self.batch_norm_fc('fc_m_bn_2', fc_m_2, activate=None)

        fc_s_1 = self.fc_layer('fc_s_1', fc_bn_1, 3000, activate=None)
        fc_s_bn_1 = self.batch_norm_fc('fc_s_bn_1', fc_s_1, activate='relu')
        fc_s_2 = self.fc_layer('fc_s_2', fc_s_bn_1, self.scateid_size, activate=None)
        fc_s_bn_2 = self.batch_norm_fc('fc_s_bn_2', fc_s_2, activate=None)


        fc_d_1 = self.fc_layer('fc_d_1', fc_bn_1, 2000, activate=None)
        fc_d_bn_1 = self.batch_norm_fc('fc_d_bn_1', fc_d_1, activate='relu')
        fc_d_2 = self.fc_layer('fc_d_2', fc_d_bn_1, self.dcateid_size, activate=None)
        fc_d_bn_2 = self.batch_norm_fc('fc_d_bn_2', fc_d_2, activate=None)

        return fc_b_bn_2, fc_m_bn_2, fc_s_bn_2, fc_d_bn_2

    def build_last_layer(self):

        last_layer_1 = self.fc_layer('last_layer_1', tf.concat([self.logits_1, self.logits_2, self.logits_3, self.logits_4, self.logits_5], 1), 6000, activate=None)
        last_layer_bn_1 = self.batch_norm_fc('last_layer_bn_1', last_layer_1, activate='relu')
        last_layer_1 = self.fc_layer('last_layer_2', last_layer_bn_1, self.num_classes, activate=None)
        last_layer_bn_2 = self.batch_norm_fc('last_layer_bn_2', last_layer_1, activate=None)
        return last_layer_bn_2

    def compute_model_1_loss(self):
        self.bcateid_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_1, labels=self.bcateid))
        self.mcateid_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_2, labels=self.mcateid))
        self.scateid_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_3, labels=self.scateid))
        self.dcateid_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_4, labels=self.dcateid))
        self.bcateid_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.logits_1, axis=1, output_type=tf.int32), self.bcateid), tf.float32),
            name='bcateid_accuracy')
        self.mcateid_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.logits_2, axis=1, output_type=tf.int32), self.mcateid), tf.float32),
            name='mcateid_accuracy')
        self.scateid_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.logits_3, axis=1, output_type=tf.int32), self.scateid), tf.float32),
            name='scateid_accuracy')
        self.dcateid_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.logits_4, axis=1, output_type=tf.int32), self.dcateid), tf.float32),
            name='dcateid_accuracy')
        # tf.losses.add_loss(self.bcateid_loss)
        # tf.losses.add_loss(self.mcateid_loss)
        # tf.losses.add_loss(self.scateid_loss)
        # tf.losses.add_loss(self.dcateid_loss)
        return tf.reduce_sum([self.bcateid_loss, self.mcateid_loss, self.scateid_loss, self.dcateid_loss])

    def compute_model_2_loss(self):
        class_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_5, labels=self.label))
        self.class_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.logits_5, axis=1, output_type=tf.int32), self.label), tf.float32),
            name='class_accuracy')
        # tf.losses.add_loss(self.class_loss)
        return class_loss

    def compute_last_layer_loss(self):
        last_class_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_6, labels=self.label))
        self.last_class_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.logits_6, axis=1, output_type=tf.int32), self.label), tf.float32),
            name='last_class_accuracy')
        # tf.losses.add_loss(self.last_class_loss)
        return last_class_loss

    def train(self, epoch, batch_idx, batch_x, input_feature, batch_y):
        if epoch < 10:
            lr = 0.001
        elif epoch < 20:
            lr = 0.0007
        elif epoch < 30:
            lr = 0.0005
        elif epoch < 40:
            lr = 0.0003
        else :
            lr = 0.0001

        # feed_dict_1 = {self.input_x: batch_x,
        #                self.input_feature: input_feature,
        #                self.bcateid: batch_y[:, 0],
        #                self.mcateid: batch_y[:, 1],
        #                self.scateid: batch_y[:, 2],
        #                self.dcateid: batch_y[:, 3],
        #                self.label: batch_y[:, 4],
        #                self.learning_rate: lr}

        # feed_dict_2 = {self.input_x: batch_x,
        #                self.input_feature: input_feature,
        #                self.label: batch_y[:, 4],
        #                self.learning_rate: lr}


        feed_dict_3 = {self.input_x: batch_x,
                     self.input_feature : input_feature,
                     self.bcateid: batch_y[:, 0],
                     self.mcateid: batch_y[:, 1],
                     self.scateid: batch_y[:, 2],
                     self.dcateid: batch_y[:, 3],
                     self.label: batch_y[:, 4],
                     self.learning_rate : lr}


        # bcateid_loss, mcateid_loss, scateid_loss, dcateid_loss, class_loss, \
        # bcateid_accuracy, mcateid_accuracy, scateid_accuracy, dcateid_accuracy ,class_accuracy, \
        # _, _, global_step, summary_str \
        #     = self.sess.run(
        #     [self.bcateid_loss, self.mcateid_loss, self.scateid_loss, self.dcateid_loss, self.class_loss,
        #      self.bcateid_accuracy, self.mcateid_accuracy, self.scateid_accuracy, self.dcateid_accuracy, self.class_accuracy,
        #      self.model_1_optimizer, self.model_2_optimizer, self.global_step, self.summary_op],
        #     feed_dict=feed_dict_1)



        last_class_loss, bcateid_loss, mcateid_loss, scateid_loss, dcateid_loss, class_loss, \
        last_class_accuracy, bcateid_accuracy, mcateid_accuracy, scateid_accuracy, dcateid_accuracy ,class_accuracy, \
        _, global_step, summary_str \
            = self.sess.run(
            [self.last_class_loss, self.bcateid_loss, self.mcateid_loss, self.scateid_loss, self.dcateid_loss, self.class_loss,
             self.last_class_accuracy, self.bcateid_accuracy, self.mcateid_accuracy, self.scateid_accuracy, self.dcateid_accuracy, self.class_accuracy,
             self.last_optimizer, self.global_step, self.summary_op],
            feed_dict=feed_dict_3)


        self.avg_b_loss += bcateid_loss
        self.avg_m_loss += mcateid_loss
        self.avg_s_loss += scateid_loss
        self.avg_d_loss += dcateid_loss
        self.avg_c_loss += class_loss
        self.avg_l_loss += last_class_loss
        self.avg_b_acc += bcateid_accuracy
        self.avg_m_acc += mcateid_accuracy
        self.avg_d_acc += scateid_accuracy
        self.avg_s_acc += dcateid_accuracy
        self.avg_c_acc += class_accuracy
        self.avg_l_acc += last_class_accuracy



        if global_step % self.summary_step == 0 and global_step != 0:

            # summary_str = self.sess.run([self.summary_op], feed_dict=feed_dict)

            self.writer.add_summary(summary_str, global_step=global_step)
            self.saver.save(self.sess, os.path.join(self.logs_dir, 'model.ckpt'), global_step=global_step)

        if global_step % 100 == 0 :
            # print('{} Epoch: {}, Step: {}, accuracy: {:.4f}, loss: {:.8f}'.format(
            #     datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, global_step, self.avg_acc/100, self.avg_loss/100))

            print('{} Epoch: {}, Step: {}, last_class_loss: {:.4f}, bcateid_loss: {:.4f}, mcateid_loss: {:.4f},'
                  ' scateid_loss: {:.4f}, dcateid_loss: {:.4f}, class_loss: {:.4f}, last_class_accuracy:{:.4f} , bcateid_accuracy: {:.4f} , mcateid_accuracy: {:.4f}'
                  ', scateid_accuracy: {:.4f}, dcateid_accuracy: {:.4f}, class_accuracy: {:.4f}'.format(
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, global_step, last_class_loss, bcateid_loss,
                mcateid_loss, scateid_loss, dcateid_loss, class_loss, last_class_accuracy, bcateid_accuracy, mcateid_accuracy, scateid_accuracy,
                dcateid_accuracy, class_accuracy))
            # print('{} Epoch: {}, Step: {}, bcateid_loss: {:.4f}, mcateid_loss: {:.4f},'
            #       ' scateid_loss: {:.4f}, dcateid_loss: {:.4f}, class_loss: {:.4f}, bcateid_accuracy: {:.4f} , mcateid_accuracy: {:.4f}'
            #       ', scateid_accuracy: {:.4f}, dcateid_accuracy: {:.4f}, class_accuracy: {:.4f}'.format(
            #     datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, global_step, self.avg_b_loss/100,
            #     self.avg_m_loss / 100, self.avg_d_loss/100, self.avg_s_loss/100, self.avg_c_loss/100, self.avg_b_acc/100, self.avg_m_acc/100, self.avg_d_acc/100,
            #     self.avg_s_acc / 100, self.avg_c_acc/100))
            self.avg_b_loss = .0
            self.avg_m_loss = .0
            self.avg_s_loss = .0
            self.avg_d_loss = .0
            self.avg_c_loss = .0
            self.avg_l_loss = .0
            self.avg_b_acc = .0
            self.avg_m_acc = .0
            self.avg_s_acc = .0
            self.avg_d_acc = .0
            self.avg_c_acc = .0
            self.avg_l_acc = .0
            # print('{} Epoch: {}, Step: {}, last_class_loss: {:.4f}, bcateid_loss: {:.4f}, mcateid_loss: {:.4f},'
            #       ' scateid_loss: {:.4f}, dcateid_loss: {:.4f}, class_loss: {:.4f}, last_class_accuracy:{:.4f} , bcateid_accuracy: {:.4f} , mcateid_accuracy: {:.4f}'
            #       ', scateid_accuracy: {:.4f}, dcateid_accuracy: {:.4f}, class_accuracy: {:.4f}'.format(
            #     datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, global_step, last_class_loss, bcateid_loss,
            #     mcateid_loss, scateid_loss, dcateid_loss, class_loss, last_class_accuracy, bcateid_accuracy, mcateid_accuracy, scateid_accuracy,
            #     dcateid_accuracy, class_accuracy))

    def validate(self, batch_x, input_feature, batch_y):

        feed_dict = {self.input_x: batch_x,
                     self.input_feature : input_feature,
                     self.label: batch_y}
        # ox, accuracy, loss = self.sess.run([self.ox, self.accuracy, self.total_loss], feed_dict=feed_dict)
        logits, ox, loss = self.sess.run([self.logits, self.ox, self.total_loss], feed_dict=feed_dict)
        return np.argmax(logits, 1).reshape(-1), ox, loss


    def predict_last_layer(self, x, input_feature):
        # import numpy as np
        feed_dict = {self.input_x: x,
                     self.input_feature: input_feature}
        # logits_1, logits_2, logits_3, logits_4,
        logits_5, logits_6 = self.sess.run(
            # [tf.nn.softmax(self.logits_1),
            #  tf.nn.softmax(self.logits_2),
            #  tf.nn.softmax(self.logits_3),
            #  tf.nn.softmax(self.logits_4),
             [tf.nn.softmax(self.logits_5),
             tf.nn.softmax(self.logits_6), ], feed_dict=feed_dict)

        # return np.stack([np.argmax(logits_1, 1), np.max(logits_1, 1)], 1),\
        #        np.stack([np.argmax(logits_2, 1), np.max(logits_2, 1)], 1),\
        #        np.stack([np.argmax(logits_3, 1), np.max(logits_3, 1)], 1),\
        #        np.stack([np.argmax(logits_4, 1), np.max(logits_4, 1)], 1),\
        return np.stack([np.argmax(logits_5, 1), np.max(logits_5, 1)], 1),\
               np.stack([np.argmax(logits_6, 1), np.max(logits_6, 1)], 1)


if __name__ == '__main__':
    a = kakao(10, True)