import globalV
import numpy as np
from alexnet import *

class attribute (object):
    def __init__(self):
        # All placeholder
        self.x = tf.placeholder(tf.float32, name='x', shape=[None, globalV.FLAGS.height, globalV.FLAGS.width, 3])
        self.y = tf.placeholder(tf.int64, name='yLabel', shape=[None])
        self.att = tf.placeholder(tf.float32, name='Att', shape=[None, globalV.FLAGS.numAtt])
        self.isTraining = tf.placeholder(tf.bool)

        # Transform feature to same size as attribute (Linear compatibility)
        with tf.variable_scope("attribute"):
            wF_H = tf.get_variable(name='wF_H', shape=[1000, 500], dtype=tf.float32)
            bF_H = tf.get_variable(name='bF_H', shape=[500], dtype=tf.float32)
            wH_A = tf.get_variable(name='wH_A', shape=[500, globalV.FLAGS.numAtt], dtype=tf.float32)
            bH_A = tf.get_variable(name='bH_A', shape=[globalV.FLAGS.numAtt], dtype=tf.float32)
            wH_C = tf.get_variable(name='wH_C', shape=[500, globalV.FLAGS.numClass], dtype=tf.float32)
            bH_C = tf.get_variable(name='bH_C', shape=[globalV.FLAGS.numClass], dtype=tf.float32)

        # Input -> feature extract from CNN darknet
        self.coreNet = alexnet(self.x)

        # Feature -> Attribute
        self.hiddenF = tf.tanh(tf.add(tf.matmul(self.coreNet, wF_H), bF_H))
        self.outAtt = tf.add(tf.matmul(self.hiddenF, wH_A), bH_A)

        # Feature -> Classes
        self.outClass = tf.add(tf.matmul(self.hiddenF, wH_C), bH_C)

        # Loss
        # self.totalLoss = tf.reduce_mean(tf.squared_difference(self.att, self.outAtt))
        # self.totalLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.att, logits=self.outAtt))
        self.totalLoss = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(self.y, globalV.FLAGS.numClass), logits=self.outClass))

        # Accuracy
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.outClass, 1), self.y), tf.float32))

        # Define Optimizer
        updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(updateOps):
            self.trainStep = tf.train.AdamOptimizer(globalV.FLAGS.lr).minimize(self.totalLoss)

        # Log all output
        self.averageL = tf.placeholder(tf.float32)
        tf.summary.scalar("averageLoss", self.averageL)
        self.averageA = tf.placeholder(tf.float32)
        tf.summary.scalar("averageAcc", self.averageA)

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
        self.saver.restore(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/model/model.ckpt')
        npzFile = np.load(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/model/checkpoint.npz')
        self.Start = npzFile['Start']
        self.Check = npzFile['Check']

    def trainAtt(self, seenX, seenAttY, unseenX, unseenAttY, unseenX2, unseenAttY2):
        for i_0 in range(self.Start, globalV.FLAGS.maxSteps + 1):
            print('Loop {0}/{1}'.format(i_0, globalV.FLAGS.maxSteps))

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
                trainLoss, _ = self.sess.run([self.totalLoss ,self.trainStep], feed_dict={self.x: xBatchSeen, self.att: attYBatchSeen, self.isTraining: 1})
                losses.append(trainLoss)

            feed = {self.averageL: sum(losses) / len(losses)}
            summary = self.sess.run(self.merged, feed_dict=feed)
            self.trainWriter.add_summary(summary, i_0)

            # Validation
            losses = []
            for j in range(0, unseenX.shape[0], globalV.FLAGS.batchSize):
                xBatch = unseenX[j:j + globalV.FLAGS.batchSize]
                attYBatch = unseenAttY[j:j + globalV.FLAGS.batchSize]
                valLoss = self.sess.run(self.totalLoss, feed_dict={self.x: xBatch, self.att: attYBatch, self.isTraining: 0})
                losses.append(valLoss)
            feed = {self.averageL: sum(losses) / len(losses)}
            summary = self.sess.run(self.merged, feed_dict=feed)
            self.valWriter.add_summary(summary, i_0)

            # Test
            losses = []
            for j in range(0, unseenX2.shape[0], globalV.FLAGS.batchSize):
                xBatch = unseenX2[j:j + globalV.FLAGS.batchSize]
                attYBatch = unseenAttY2[j:j + globalV.FLAGS.batchSize]
                valLoss = self.sess.run(self.totalLoss, feed_dict={self.x: xBatch, self.att: attYBatch, self.isTraining: 0})
                losses.append(valLoss)
            feed = {self.averageL: sum(losses) / len(losses)}
            summary = self.sess.run(self.merged, feed_dict=feed)
            self.testWriter.add_summary(summary, i_0)

            if i_0 % self.Check == 0:
                savePath = self.saver.save(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/model/model.ckpt',global_step=i_0)
                print('Model saved in file: {0}'.format(savePath))
                if i_0 / self.Check == 10:
                    self.Check *= 10

            # Save State
            np.savez(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/model/checkpoint.npz', Start=i_0+1, Check=self.Check)
            self.saver.save(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/model/model.ckpt')

    def trainClass(self, x70, y70, x30, y30):
        for i_0 in range(self.Start, globalV.FLAGS.maxSteps + 1):
            print('Loop {0}/{1}'.format(i_0, globalV.FLAGS.maxSteps))

            # Shuffle Data
            s = np.arange(x70.shape[0])
            np.random.shuffle(s)
            x70 = x70[s]
            y70 = y70[s]

            # Train
            losses = []
            accs = []
            for j in range(0, x70.shape[0], globalV.FLAGS.batchSize):
                xBatchSeen = x70[j:j + globalV.FLAGS.batchSize]
                yBatchSeen = y70[j:j + globalV.FLAGS.batchSize]
                trainLoss, trainAcc, _ = self.sess.run([self.totalLoss , self.accuracy, self.trainStep], feed_dict={self.x: xBatchSeen, self.y: yBatchSeen, self.isTraining: 1})
                losses.append(trainLoss)
                accs.append(trainAcc)

            feed = {self.averageL: sum(losses) / len(losses), self.averageA: sum(accs) / len(accs)}
            summary = self.sess.run(self.merged, feed_dict=feed)
            self.trainWriter.add_summary(summary, i_0)

            # Test
            losses = []
            accs = []
            for j in range(0, x30.shape[0], globalV.FLAGS.batchSize):
                xBatch = x30[j:j + globalV.FLAGS.batchSize]
                yBatch = y30[j:j + globalV.FLAGS.batchSize]
                valLoss, valAcc = self.sess.run([self.totalLoss, self.accuracy], feed_dict={self.x: xBatch, self.y: yBatch, self.isTraining: 0})
                losses.append(valLoss)
                accs.append(valAcc)
            feed = {self.averageL: sum(losses) / len(losses), self.averageA: sum(accs) / len(accs)}
            summary = self.sess.run(self.merged, feed_dict=feed)
            self.testWriter.add_summary(summary, i_0)

            if i_0 % self.Check == 0:
                savePath = self.saver.save(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/model/model.ckpt',global_step=i_0)
                print('Model saved in file: {0}'.format(savePath))
                if i_0 / self.Check == 10:
                    self.Check *= 10

            # Save State
            np.savez(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/model/checkpoint.npz', Start=i_0+1, Check=self.Check)
            self.saver.save(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/model/model.ckpt')


    def getEmbedded(self, trainX):
        outputAtt = None
        for j in range(0, trainX.shape[0], globalV.FLAGS.batchSize):
            xBatch = trainX[j:j + globalV.FLAGS.batchSize]
            if outputAtt is None:
                outputAtt = self.sess.run(self.hiddenF, feed_dict={self.x: xBatch, self.isTraining: 0})
            else:
                outputAtt = np.concatenate((outputAtt, self.sess.run(self.hiddenF, feed_dict={self.x: xBatch, self.isTraining: 0})), axis=0)
        return outputAtt


