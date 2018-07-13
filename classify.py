import tensorflow as tf
import numpy as np
import globalV


class classify (object):
    def __init__(self):
        # All placeholder
        self.y = tf.placeholder(tf.int64, name='outputClassIndexSeen_C', shape=[None])
        self.att = tf.placeholder(tf.float32, name='outputClassAttSeen_C', shape=[None, globalV.FLAGS.numAtt])
        self.keepProb = tf.placeholder(tf.float32)

        # From attribute to hidden
        wH = tf.get_variable(name='wH_C', shape=[globalV.FLAGS.numAtt, 100], dtype=tf.float32)
        bH = tf.get_variable(name='bH_C', shape=[100], dtype=tf.float32)

        # From hidden to softmax
        wS = tf.get_variable(name='wS_C', shape=[100, globalV.FLAGS.numClass], dtype=tf.float32)
        bS = tf.get_variable(name='bS_C', shape=[globalV.FLAGS.numClass], dtype=tf.float32)

        # Attribute -> Hidden
        outHid = tf.sigmoid(tf.add(tf.matmul(self.att, wH), bH))
        dropOut = tf.nn.dropout(outHid, self.keepProb)

        # Hidden -> Softmax
        self.outSoft = tf.add(tf.matmul(dropOut, wS), bS)

        # Softmax Loss for seen class
        self.lossSoft = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(self.y, globalV.FLAGS.numClass), logits=self.outSoft))

        # Total Loss
        self.totalLoss = self.lossSoft

        # Define Optimizer
        updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(updateOps):
            self.trainStep = tf.train.AdamOptimizer(globalV.FLAGS.lr).minimize(self.totalLoss)

        # Calculate accuracy
        self.predictIndex = tf.argmax(self.outSoft, 1)
        self.correctPrediction = tf.equal(self.predictIndex, self.y)
        self.accuracy = tf.reduce_mean(tf.cast(self.correctPrediction, tf.float32))

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
        if globalV.FLAGS.TC == 1:
            if tf.gfile.Exists(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/classify/logs'):
                tf.gfile.DeleteRecursively(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/classify/logs')
            tf.gfile.MakeDirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/classify/logs')
        self.trainWriter = tf.summary.FileWriter(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/classify/logs/train', self.sess.graph)
        self.testWriter = tf.summary.FileWriter(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/classify/logs/test')

        # Start point and log
        self.Start = 1
        self.Check = 10
        if globalV.FLAGS.TC == 0:
            self.restoreModel()

    def restoreModel(self):
        self.saver.restore(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/classify/model/model.ckpt')
        npzFile = np.load(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/classify/model/checkpoint.npz')
        self.Start = npzFile['Start']
        self.Check = npzFile['Check']

    def trainClassify(self, att, y, keep=0.5):
        for i in range(self.Start, globalV.FLAGS.maxSteps + 1):
            if globalV.FLAGS.HEADER == 1:
                print('Loop {0}/{1}'.format(i, globalV.FLAGS.maxSteps))
            # Train
            losses = []
            accuracies = []
            for j in range(0, att.shape[0], globalV.FLAGS.batchSize):
                xBatch = att[j:j + globalV.FLAGS.batchSize]
                yBatch = y[j:j + globalV.FLAGS.batchSize]
                trainLoss, trainAccuracy, _ = self.sess.run([self.totalLoss, self.accuracy ,self.trainStep], feed_dict={self.y: yBatch, self.att: xBatch, self.keepProb: keep})
                losses.append(trainLoss)
                accuracies.append(trainAccuracy)
            feed = {self.averageL: sum(losses) / len(losses), self.averageA: sum(accuracies) / len(accuracies)}
            summary = self.sess.run(self.merged, feed_dict=feed)
            self.trainWriter.add_summary(summary, i)

            if i % self.Check == 0:
                savePath = self.saver.save(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/classify/model/model.ckpt',global_step=i)
                # print('Model saved in file: {0}'.format(savePath))
                if i / self.Check == 10:
                    self.Check *= 10

            # Save State
            np.savez(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/classify/model/checkpoint.npz', Start=i+1, Check=self.Check)
            self.saver.save(self.sess, globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/classify/model/model.ckpt')
        print(sum(losses) / len(losses), sum(accuracies) / len(accuracies))

    def predict(self, att):
        return self.sess.run(self.predictIndex, feed_dict={self.att: att, self.keepProb: 1.0})

    def predictScore(self, att):
        return self.sess.run(self.outSoft, feed_dict={self.att: att, self.keepProb: 1.0})


