import tensorflow as tf
import numpy as np
import globalV
from alexnet import *


class attribute (object):
    def __init__(self):
        # All placeholder
        self.x = tf.placeholder(tf.float32, name='inputImage', shape=[None, globalV.FLAGS.height, globalV.FLAGS.width, 3])
        self.ySeen = tf.placeholder(tf.int64, name='outputClassIndexSeen', shape=[None])
        self.attYSeen = tf.placeholder(tf.float32, name='outputClassAttSeen', shape=[None, globalV.FLAGS.numAtt])
        self.yUnseen = tf.placeholder(tf.int64, name='outputClassIndexUnseen', shape=[None])
        self.attYUnseen = tf.placeholder(tf.float32, name='outputClassAttUnseen', shape=[None, globalV.FLAGS.numAtt])
        self.isTraining = tf.placeholder(tf.bool)

        # Transform feature to same size as attribute (Linear compatibility)
        with tf.variable_scope("attribute"):
            wF_H = tf.get_variable(name='wF_H', shape=[1000, 500], dtype=tf.float32)
            bF_H = tf.get_variable(name='bF_H', shape=[500], dtype=tf.float32)
            self.wH_A = tf.get_variable(name='wH_A', shape=[500, globalV.FLAGS.numAtt], dtype=tf.float32)
            bH_A = tf.get_variable(name='bH_A', shape=[globalV.FLAGS.numAtt], dtype=tf.float32)

        # Input -> feature extract from CNN darknet
        # processX = tf.subtract(tf.multiply(tf.divide(self.x, 255), 2.0), 1.0)
        self.coreNet = alexnet(self.x)

        self.softmax = tf.nn.softmax(self.coreNet)

        # Feature -> Attribute
        hiddenF = tf.tanh(tf.add(tf.matmul(self.coreNet, wF_H), bF_H))
        self.outAtt = tf.add(tf.matmul(hiddenF, self.wH_A), bH_A)
        self.outAttSig = tf.sigmoid(self.outAtt)

        # Loss
        self.totalLoss = tf.reduce_mean(tf.squared_difference(self.attYSeen, self.outAttSig))
        # self.totalLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.attYSeen, logits=self.outAtt))

        # Define Optimizer
        updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(updateOps):
            self.trainStep = tf.train.AdamOptimizer(globalV.FLAGS.lr).minimize(self.totalLoss)

        # Log all output
        self.averageL = tf.placeholder(tf.float32)
        tf.summary.scalar("averageLoss", self.averageL)

        # Merge all log
        self.merged = tf.summary.merge_all()

        # Initialize session
        tfConfig = tf.ConfigProto(allow_soft_placement=True)
        tfConfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfConfig)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=None)

        # Log directory
        if globalV.FLAGS.TA == 1:
            if tf.gfile.Exists(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/logs'):
                tf.gfile.DeleteRecursively(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/logs')
            tf.gfile.MakeDirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/logs')
        self.trainWriter = tf.summary.FileWriter(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/logs/train', self.sess.graph)
        self.valWriter = tf.summary.FileWriter(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/logs/validate')
        self.testWriter = tf.summary.FileWriter(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/logs/test')

        # Start point and log
        self.Start = 1
        self.Check = 10
        if globalV.FLAGS.TA == 0:
            self.restoreModel()

    def restoreModel(self):
        self.saver.restore(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/model/model.ckpt-80')
        npzFile = np.load(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/model/checkpoint.npz')
        self.Start = npzFile['Start']
        self.Check = npzFile['Check']

    def trainAtt(self, seenX, seenAttY, unseenX, unseenAttY, unseenX2, unseenAttY2):
        for i in range(self.Start, globalV.FLAGS.maxSteps + 1):
            print('Loop {0}/{1}'.format(i, globalV.FLAGS.maxSteps))

            # Shuffle Data
            s = np.arange(seenX.shape[0])
            np.random.shuffle(s)
            seenX = seenX[s]
            seenAttY = seenAttY[s]

            # Train
            losses = []
            for j in range(0, seenX.shape[0], globalV.FLAGS.batchSize):
                xBatchSeen = seenX[j:j + globalV.FLAGS.batchSize]
                attYBatchSeen = seenAttY[j:j + globalV.FLAGS.batchSize]
                trainLoss, _ = self.sess.run([self.totalLoss ,self.trainStep], feed_dict={self.x: xBatchSeen, self.attYSeen: attYBatchSeen, self.isTraining: 1})
                losses.append(trainLoss)

            feed = {self.averageL: sum(losses) / len(losses)}
            summary = self.sess.run(self.merged, feed_dict=feed)
            self.trainWriter.add_summary(summary, i)

            # Validation
            losses = []
            for j in range(0, unseenX.shape[0], globalV.FLAGS.batchSize):
                xBatch = unseenX[j:j + globalV.FLAGS.batchSize]
                attYBatch = unseenAttY[j:j + globalV.FLAGS.batchSize]
                valLoss = self.sess.run(self.totalLoss, feed_dict={self.x: xBatch, self.attYSeen: attYBatch, self.isTraining: 0})
                losses.append(valLoss)
            feed = {self.averageL: sum(losses) / len(losses)}
            summary = self.sess.run(self.merged, feed_dict=feed)
            self.valWriter.add_summary(summary, i)

            # Test
            losses = []
            for j in range(0, unseenX2.shape[0], globalV.FLAGS.batchSize):
                xBatch = unseenX2[j:j + globalV.FLAGS.batchSize]
                attYBatch = unseenAttY2[j:j + globalV.FLAGS.batchSize]
                valLoss = self.sess.run(self.totalLoss, feed_dict={self.x: xBatch, self.attYSeen: attYBatch, self.isTraining: 0})
                losses.append(valLoss)
            feed = {self.averageL: sum(losses) / len(losses)}
            summary = self.sess.run(self.merged, feed_dict=feed)
            self.testWriter.add_summary(summary, i)

            if i % self.Check == 0:
                savePath = self.saver.save(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/model/model.ckpt',global_step=i)
                print('Model saved in file: {0}'.format(savePath))
                if i / self.Check == 10:
                    self.Check *= 10

            # Save State
            np.savez(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/model/checkpoint.npz', Start=i+1, Check=self.Check)
            self.saver.save(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/model/model.ckpt')

    def getAttribute(self, trainX):
        outputAtt = None
        for j in range(0, trainX.shape[0], globalV.FLAGS.batchSize):
            xBatch = trainX[j:j + globalV.FLAGS.batchSize]
            if outputAtt is None:
                outputAtt = self.sess.run(self.outAttSig, feed_dict={self.x: xBatch, self.isTraining: 0})
            else:
                outputAtt = np.concatenate((outputAtt, self.sess.run(self.outAttSig, feed_dict={self.x: xBatch, self.isTraining: 0})), axis=0)
        return outputAtt

    def getLastWeight(self):
        return self.sess.run(self.wH_A)

    def getSoftMax(self, xInput):
        return self.sess.run(self.softmax, feed_dict={self.x: xInput})

