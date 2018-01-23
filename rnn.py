#coding:utf-8

import tensorflow as tf
from tensorflow.contrib import rnn
import pandas as pd
import numpy as np
import random, time, math, sys, progressbar
sys.path.insert(0,'..')
from inc import prices, toTS, fromTS, getLast
from tensorflow.python import debug as tf_debug
from sklearn import preprocessing
#import matplotlib.pyplot as plt


def timeNow():
    return(int(str(time.time()).split('.')[0]))

def writePred(pred):
    f = open('../pred', 'a')
    f.write(str(pred[0][0]) + ',' + str(pred[0][1])+ ',' + str(pred[0][2]) + '\n')

class RNN(object):

    def __init__(self, sess, model, abc, xyz, starter_learning_rate, decay, iters, display_step, batch, win, future, backward, n_hidden, n_layers, n_states, wn, writer):
        self.sess = sess
        self.model = model
        self.abc = abc
        self.xyz = xyz
        self.starter_learning_rate = starter_learning_rate
        self.decay = decay
        self.iters = iters
        self.display_step = display_step
        self.batch = batch
        self.win = win
        self.future = future
        self.backward = backward
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_states = n_states
        self.wn = wn
        self.writer = writer

        self.graf()

    def placeNum(self, nums):

        arr = []

        for x in nums:
            if x <= -0.15: arr.append(0)
            elif x >= 0.15: arr.append(2)
            else: arr.append(1)

        return arr

    def getCandles(self):
        start = toTS(self.abc)
        finish = toTS(self.xyz)
        pr, times = prices(start * 1000000000, finish * 1000000000, self.win, 'rts')
        mn = (len(pr) // self.backward) * self.backward
        pr = pr[0:mn]
        #print(pr)
        pr = np.reshape(pr, [-1, self.backward])
        pr = preprocessing.normalize(pr)
        x = np.round(pr, decimals=3)
        #x = np.round(np.round(pr * 3) / 3, decimals=1)
        #y = np.round(pr * 10)
        y = pr

        """plt.plot(times, x)
        plt.show()
        plt.close()
        plt.plot(times, y)
        plt.show()
        plt.close()"""

        x = np.reshape(x, -1)
        y = np.reshape(y, -1)

        #unique, counts = np.unique(x, return_counts=True)
        #print(dict(zip(unique, counts)), len(unique))
        y = self.placeNum(y)
        unique, counts = np.unique(y, return_counts=True)
        print(dict(zip(unique, counts)), len(unique))

        return x, y

    def prepare(self, _x, _y, start):

        x=[]
        y=[]

        i = 0
        for _ in range(self.batch):
            """one_predictor=_x[start+i*self.batch:start+i*self.batch+self.backward]
            x.append(one_predictor)
            one_factor=_y[start+i*self.batch + self.future:start+i*self.batch + self.backward + self.future]
            y.append(one_factor)
            i += 1"""
            one_predictor=_x[start+i:start+i+self.backward]
            x.append(one_predictor)
            one_factor=_y[start+i+self.future:start+i+self.backward+self.future]
            y.append(one_factor)
            i += 1
        x = np.reshape(x, [self.batch, self.backward, 1])
        y = np.reshape(y, [self.batch, self.backward])

        return x, y

    def extract_axis_1(self, tensor, index):
        batch_range = tf.range(0, self.batch)
        index_range = batch_range * 0 + index
        indexes = tf.stack([batch_range, index_range], axis=1)
        values = tf.gather_nd(tensor, indexes)
        res = tf.reshape(values, [self.batch, 1])
        return res

    def decay_lr(self, start, decay, gs):
        lr = start*(decay**gs)
        #if lr >= 0.0001:
        #    return lr
        #else:
        #    return 0.0001
        return lr

    def restate(self):
        with tf.variable_scope('states', reuse=True):
            restored = []
            for i in range(self.n_layers):
                c = self.sess.run(tf.get_variable('ss' + str(i) + 'c'))
                h = self.sess.run(tf.get_variable('ss' + str(i) + 'h'))
                restored.append(tf.contrib.rnn.LSTMStateTuple(c, h))
        return tuple(restored)


    def graf(self):

        self.gs = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int32)
        self.lr = tf.Variable(self.starter_learning_rate, trainable=False, name='lerning_rate', dtype=tf.float32)

        with tf.variable_scope('states'):
            for i in range(self.n_layers):
                tf.get_variable('ss' + str(i) + 'c', [self.batch, self.n_hidden])
                tf.get_variable('ss' + str(i) + 'h', [self.batch, self.n_hidden])

        with tf.name_scope('RNNvar'):
            WR = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[self.n_hidden, self.n_states]), name='WR')
            tf.summary.histogram('WR', WR)
            BR = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[self.n_states]), name='BR')
            tf.summary.histogram('BR', BR)

        with tf.name_scope('CNNvar'):
            WC1 = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[64, 1024]), name='WC1')
            tf.summary.histogram('WC1', WC1)
            BC1 = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[1024]), name='BC1')
            tf.summary.histogram('BC1', BC1)
            WC2 = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[1024, self.backward]), name='WC2')
            tf.summary.histogram('WC2', WC2)
            BC2 = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[self.backward]), name='BC2')
            tf.summary.histogram('BC2', BC2)

        self.X = tf.placeholder(name='X', dtype=tf.float32, shape=[self.batch, self.backward, 1])
        self.Y = tf.placeholder(name='Y', dtype=tf.int64, shape=[self.batch, self.backward])

        cell = tf.contrib.rnn.LSTMBlockCell(self.n_hidden, forget_bias=0.0)
        if self.wn =='train' or self.wn == 'cont':
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.75)
        cells = tf.contrib.rnn.MultiRNNCell([cell for _ in range(self.n_layers)])

        self.initial_state = cells.zero_state(self.batch, tf.float32)

        with tf.name_scope('RNN'):
            outputs, state = tf.nn.dynamic_rnn(cells, self.X, initial_state=self.initial_state, dtype=tf.float32)
            tf.summary.histogram('output', outputs)
            self.final_state = state
            x = tf.reshape(tf.concat(outputs, 1), [-1, self.n_hidden])
            y = tf.matmul(x, WR) + BR
            tf.summary.histogram('YYY', y)

        seq_y = tf.reshape(y, [self.batch, self.backward, self.n_states])

        rnn_cnn = tf.reshape(seq_y, [-1, self.backward, self.n_states, 1])

        with tf.name_scope('CNN'):
            conv1 = tf.layers.conv2d(inputs=rnn_cnn, filters=32, kernel_size=[5, 5], padding="same")
            tf.summary.histogram('conv1', conv1)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=1, strides=1)
            tf.summary.histogram('pool1', pool1)
            conv2 = tf.layers.conv2d( inputs=pool1, filters=64, kernel_size=[5, 5], padding="same")
            tf.summary.histogram('conv2', conv2)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[self.backward, self.n_states], strides=1)
            tf.summary.histogram('pool2', pool2)
            pool2_flat = tf.reshape(pool2, [-1, 64])
            pred_predict = tf.matmul(pool2_flat, WC1) + BC1
            tf.summary.histogram('pred_predict', pred_predict)
            if self.wn == 'train':
                 pred_predict = tf.nn.dropout(pred_predict, 0.75)
                 tf.summary.histogram('drop', pred_predict)
            self.predict = tf.matmul(pred_predict, WC2) + BC2
            tf.summary.histogram('predict', self.predict)

        self.seq = tf.argmax(seq_y, 2)

        last_y = self.extract_axis_1(self.Y, self.backward - 1)
        last_r = self.extract_axis_1(self.seq, self.backward - 1)
        last_c = self.extract_axis_1(self.predict, self.backward - 1)

        with tf.name_scope('loose'):
            self.loose = tf.contrib.seq2seq.sequence_loss(seq_y, self.Y, tf.ones([self.batch, self.backward], tf.float32), average_across_timesteps=True, average_across_batch=False)
            self.loose = tf.reduce_mean(self.loose)
            tf.summary.scalar('rnn', self.loose)
            #loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.predict, labels=self.Y)
            loss = tf.losses.huber_loss(self.Y, self.predict)
            #loss = tf.losses.absolute_difference(self.Y, self.predict)
            loss = tf.reduce_mean(loss)
            tf.summary.scalar('cnn', loss)

        with tf.name_scope('accuracy'):
            correct_prediction_rnn = tf.equal(self.Y, self.seq)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction_rnn, tf.float32))
            tf.summary.scalar('rnn', self.accuracy)
            correct_prediction_cnn = tf.equal(self.Y, tf.to_int64(tf.round(self.predict)))
            self.precise = tf.reduce_mean(tf.cast(correct_prediction_cnn, tf.float32))
            tf.summary.scalar('cnn', self.precise)

        with tf.name_scope('dif'):
            self.dif = tf.reduce_max(tf.losses.absolute_difference(last_y, last_r))
            tf.summary.scalar('rnn', self.dif)
            self.diff = tf.reduce_max(tf.losses.absolute_difference(last_y, last_c))
            tf.summary.scalar('cnn', self.diff)

        with tf.name_scope('train'):
            var_r = [WR, BR]
            var_c = [WC1, BC1, WC2, BC2]
            grad_r, _ = tf.clip_by_global_norm(tf.gradients(self.loose, var_r), 5)
            grad_c, _ = tf.clip_by_global_norm(tf.gradients(loss, var_c), 5)
            optimizer = tf.train.AdamOptimizer(self.lr)
            train_r = optimizer.apply_gradients(zip(grad_r, var_r), name='TrainRNN')
            train_c = optimizer.apply_gradients(zip(grad_c, var_c), name='TrainCNN')
            self.train = tf.group(train_r, train_c)

        self.merged = tf.summary.merge_all()

        self.saver = tf.train.Saver()

        #self.sup = pred_predict
        self.sup = tf.shape(1)

    def run(self):
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        if self.wn == 'train':
            init = tf.global_variables_initializer()
            self.sess.run(init)
            _initial_state = self.sess.run(self.initial_state)
        else:
            self.saver.restore(self.sess, 'models/' + self.model + '/mod.ckpt')
            _initial_state = self.restate()
        if self.wn == 'cont' or self.wn == 'train':

            if self.writer:
                train_writer = tf.summary.FileWriter('logs/train' + str(timeNow()), self.sess.graph)
                test_writer = tf.summary.FileWriter('logs/test' + str(timeNow()))

            get_x, get_y = self.getCandles()
            divide = len(get_x) // 10
            train_x = get_x[0:divide*9]
            train_y = get_y[0:divide*9]
            test_x = get_x[divide*9:-1]
            test_y = get_y[divide*9:-1]
            in_train = len(train_x) - self.future
            train_batch = in_train // self.batch
            in_test = len(test_x) - self.future
            test_batch = in_test // self.batch

            print('ALL:' , len(get_x), '\nTRAIN: ', in_train, 'BATCH: ', train_batch, '\nTEST: ', in_test, 'BATCH: ', test_batch)
            step = 1
            for j in range(self.iters):
                tt = time.time()
                for i in range(train_batch):
                    _x, _y = self.prepare(train_x, train_y, i * self.batch)
                    #print(len(_x), len(_y))
                    dic={self.X: _x, self.Y: _y, self.initial_state: _initial_state}
                    _, _s, _p, _final_state, _l, _a, summary  = self.sess.run([self.train, self.seq, self.predict, self.final_state, self.loose, self.accuracy, self.merged], feed_dict=dic)
                    _initial_state = _final_state
                    if self.writer and j > 0: train_writer.add_summary(summary, step)
                    step += 1
                if j % self.display_step == 0:
                    rnd = random.randint(0, self.batch - 5)
                    i_rnd = random.randint(0, test_batch -1)
                    __x, __y = self.prepare(test_x, test_y, i_rnd)
                    dic = {self.X: __x, self.Y: __y}
                    __sup, __s, __p, __l, __a, __lr, _gs, summary = self.sess.run([self.sup, self.seq, self.predict, self.loose, self.accuracy, self.lr, self.gs, self.merged], feed_dict=dic)
                    if self.writer: test_writer.add_summary(summary, step)
                    for i in range(len(__y)):
                        print(_y[i])
                        print(_s[i])
                        print(np.round(np.maximum(_p[i], 0)).astype(int))
                        print('------------')
                    print('\nepoch: %s | iter: %s \
                          \nloss: %s | acc: %s \
                          \nY: %s \
                          \ny: %s \
                          \np: %s \
                          \nloss: %s | acc: %s | lr: %s \
                          '%(_gs, step, _l, _a, __y[rnd], __s[rnd], np.maximum(np.round(np.maximum(__p[rnd], 0)).astype(int), 0), __l, __a, __lr))
                    print('+++\n', __sup, '\n+++')
                    print(time.time() - tt)
                with tf.variable_scope('states', reuse=True):
                    for i, (c, h) in enumerate(_final_state):
                        ss = tf.get_variable('ss' + str(i) + 'c')
                        tf.assign(ss, c)
                        ss = tf.get_variable('ss' + str(i) + 'h')
                        tf.assign(ss, h)
                _lr = self.decay_lr(self.starter_learning_rate, self.decay, _gs)
                self.sess.run(tf.assign(self.lr, _lr))
                self.sess.run(tf.assign(self.gs, self.gs+1))
                self.saver.save(self.sess, 'models/' + self.model + '/mod.ckpt')
            print('Optimization Finished!')

    def test(self):

        x, y = self.getCandles()
        #print(y)
        #print(max(x))
        #print(min(x))
        #print(x)

        """bub = bub[0:6000]
        bub = np.reshape(bub, [-1, 20])

        bub = np.round(preprocessing.normalize(bub), decimals=3)
        print(bub)
        unique, counts = np.unique(bub, return_counts=True)
        print(dict(zip(unique, counts)), len(unique))"""

        var1 = tf.shape(1)
        var2 = tf.shape(1)
        var3 = tf.shape(1)
        var4 = tf.shape(1)

        for i in range(100):
            _x, _y = self.prepare(x, y, i)
            print(_x)
            print(_y)
            time.sleep(2)
        dic={self.X: _x, self.Y: _y}
        var1, var2, var3, var4 = self.sess.run([var1, var2, var3, var4], feed_dict=dic)
        print(var1)
        print(var2)
        print(var3)
        print(var4)