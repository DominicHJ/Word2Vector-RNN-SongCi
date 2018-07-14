#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class Model():
    def __init__(self, learning_rate=0.001, batch_size=16, num_steps=32, num_words=5000, dim_embedding=128,
                 rnn_layers=3):
        r"""初始化函数

        Parameters
        ----------
        learning_rate : float
            学习率.
        batch_size : int
            batch_size.
        num_steps : int
            RNN有多少个time step，也就是输入数据的长度是多少.
        num_words : int
            字典里有多少个字，用作embeding变量的第一个维度的确定和onehot编码.
        dim_embedding : int
            embding中，编码后的字向量的维度
        rnn_layers : int
            有多少个RNN层，在这个模型里，一个RNN层就是一个RNN Cell，各个Cell之间通过TensorFlow提供的多层RNNAPI（MultiRNNCell等）组织到一起

        """
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_words = num_words
        self.dim_embedding = dim_embedding
        self.rnn_layers = rnn_layers
        self.learning_rate = learning_rate

    def build(self, embedding_file=None):
        # global step
        self.global_step = tf.Variable(
            0, trainable=False, name='self.global_step', dtype=tf.int64)

        self.X = tf.placeholder(
            tf.int32, shape=[None, self.num_steps], name='input')
        self.Y = tf.placeholder(
            tf.int32, shape=[None, self.num_steps], name='label')

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.variable_scope('embedding'):
            if embedding_file:
                # if embedding file provided, use it.
                embedding = np.load(embedding_file)
                embed = tf.constant(embedding, name='embedding')
            else:
                # if not, initialize an embedding and train it.
                embed = tf.get_variable(
                    'embedding', [self.num_words, self.dim_embedding])
                tf.summary.histogram('embed', embed)

            data = tf.nn.embedding_lookup(embed, self.X)

        def get_a_cell(lstm_size, keep_prob):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop

        with tf.variable_scope('rnn'):
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_a_cell(self.dim_embedding, self.keep_prob) for _ in range(self.rnn_layers)]
            )

            self.state_tensor = cell.zero_state(self.batch_size, tf.float32)
            outputs_tensor, final_state = tf.nn.dynamic_rnn(cell, data, initial_state=self.state_tensor)
            self.outputs_state_tensor = final_state

            # concate every time step
        seq_output = tf.concat(outputs_tensor, 1)

        # flatten it
        seq_output_final = tf.reshape(tf.concat(outputs_tensor, 1), [-1, self.dim_embedding])
        print('seq_output.shape: ', seq_output.shape)
        print('seq_output_final.shape: ', seq_output_final.shape)

        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [self.dim_embedding, self.num_words])
            b = tf.get_variable('b', [self.num_words])

        logits = tf.matmul(seq_output_final, W) + b

        tf.summary.histogram('logits', logits)

        self.predictions = tf.nn.softmax(logits, name='predictions')

        y_one_hot = tf.one_hot(self.Y, self.num_words)
        print('y_one_hot.shape: ', y_one_hot.shape)
        y_reshaped = tf.reshape(y_one_hot, logits.get_shape())

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_reshaped, logits=logits)
        mean, var = tf.nn.moments(logits, -1)
        self.loss = tf.reduce_mean(loss)
        tf.summary.scalar('logits_loss', self.loss)

        var_loss = tf.divide(10.0, 1.0 + tf.reduce_mean(var))
        tf.summary.scalar('var_loss', var_loss)
        self.loss = self.loss + var_loss
        tf.summary.scalar('total_loss', self.loss)

        # gradient clip
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(
            zip(grads, tvars), global_step=self.global_step)

        tf.summary.scalar('loss', self.loss)

        self.merged_summary_op = tf.summary.merge_all()
