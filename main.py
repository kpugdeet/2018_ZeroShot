import numpy as np
import tensorflow as tf
import argparse
import os
import plotly.plotly as py
import plotly.graph_objs as go
from scipy import spatial
from graphviz import Digraph

"""
import created files
"""
import globalV
from loadData import loadData
from attribute import attribute
from classify import classify

"""
Disable unnecessary logs.
Login to plotting account.
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
py.sign_in('krittaphat.pug', 'oVTAbkhd2RQvodGOwrwp') # G-mail Login
# py.sign_in('amps1', 'Z1KAk8xiPUyO2U58JV2K') # kpugdeet@syr.edu/12345678
# py.sign_in('amps2', 'jGQHMBArdACog36YYCAI') # yli41@syr.edu/12345678
# py.sign_in('amps3', '5geLaNJlswmzDucmKikR') # liyilan0120@gmail.com/12345678

class dataClass:
    """
    Data class to keep all data to train and test model
    """
    trX70 = None
    trY70 = None
    trAtt70 = None
    trX30 = None
    trY30 = None
    trAtt30 = None
    vX = None
    vY = None
    vAtt = None
    teX = None
    teY = None
    teAtt = None
    baX70 = None
    baAtt70 = None
    concatAtt = None
    allClassName = None
    treeMap = None
    distanceFunc = None
    tmpConcatAtt = None

def argumentParser():
    """
    Get all argument values from command line.
    :return: No return, set value to globalV variable.
    """
    parser = argparse.ArgumentParser()

    # Input image size & attribute dimension
    parser.add_argument('--width', type=int, default=227, help='Image width')
    parser.add_argument('--height', type=int, default=227, help='Image height')
    parser.add_argument('--SELATT', type=int, default=0, help='0.Att, 1.Word2Vec, 2.Att+Word2Vec')
    parser.add_argument('--numAtt', type=int, default=64, help='Dimension of Attribute')

    # Dataset Path
    parser.add_argument('--BASEDIR', type=str, default='/media/dataHD3/kpugdeet/', help='Base folder for dataset and logs')
    parser.add_argument('--AWA2PATH', type=str, default='AWA2/Animals_with_Attributes2/', help='AWA2 dataset')
    parser.add_argument('--CUBPATH', type=str, default='CUB/CUB_200_2011/', help='CUB dataset')
    parser.add_argument('--SUNPATH', type=str, default='SUN/SUNAttributeDB/', help='SUN dataset')
    parser.add_argument('--APYPATH', type=str, default='APY/attribute_data/', help='APY dataset')
    parser.add_argument('--GOOGLE', type=str, default='PRE/GoogleNews-vectors-negative300.bin', help='Google Word2Vec model')

    # Working directory
    parser.add_argument('--KEY', type=str, default='APY', help='Choose dataset (AWA2, CUB, SUN, APY)')
    parser.add_argument('--DIR', type=str, default='APY_0', help='Choose working directory')
    parser.add_argument('--numClass', type=int, default=32, help='Number of class')

    # Hyper Parameter
    parser.add_argument('--maxSteps', type=int, default=0, help='Number of steps to run trainer.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--batchSize', type=int, default=32, help='Batch size')

    # Initialize or Restore Model
    parser.add_argument('--TD', type=int, default=0, help='0.Restore 1.Train visual model')
    parser.add_argument('--TA', type=int, default=0, help='0.Restore 1.Train attribute model')
    parser.add_argument('--TC', type=int, default=0, help='0.Restore 1.Train classify model')

    # Choose what to do
    parser.add_argument('--OPT', type=int, default=0, help='0.CNN, 1.Attribute, 2.Classify, 3.Accuracy')

    # ETC.
    parser.add_argument('--HEADER', type=int, default=0, help='0.Not-Show, 1.Show')
    parser.add_argument('--SEED', type=int, default=0, help='Random number for shuffle data')

    globalV.FLAGS, _ = parser.parse_known_args()

def createFolder():
    """
    Check Folder exist.
    :return: None
    """
    if not os.path.exists(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR):
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR)
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup')
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute')
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/logs')
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/attribute/model')
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/classify')
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/classify/logs')
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/classify/model')

def loadDataset():
    """
    Load data set and split to Train, Validatio, and Test set.
    Choose whether to use attribute, word2Vec, or attribute+word2Vec to train attribute and classify model.

    :return:
    """
    (trainClass, trainAtt, trainVec, trainX, trainY, trainYAtt), \
    (valClass, valAtt, valVec, valX, valY, valYAtt), \
    (testClass, testAtt, testVec, testX, testY, testYAtt) = loadData.getData()

    if globalV.FLAGS.HEADER == 1:
        print('\nLoad Data for {0}'.format(globalV.FLAGS.KEY))
        if globalV.FLAGS.KEY == 'SUN' or globalV.FLAGS.KEY == 'APY':
            print('       {0:>10} {1:>12} {2:>10} {3:>20} {4:>10} {5:>12}'.format('numClass', 'classAtt', 'classVec', 'inputX', 'outputY', 'outputAtt'))
            print('Train: {0:>10} {1:>12} {2:>10} {3:>20} {4:>10} {5:>12}'.format(len(trainClass), str(trainAtt.shape), str(trainVec.shape), str(trainX.shape), str(trainY.shape), str(trainYAtt.shape)))
            print('Valid: {0:>10} {1:>12} {2:>10} {3:>20} {4:>10} {5:>12}'.format(len(valClass), str(valAtt.shape), str(valVec.shape), str(valX.shape), str(valY.shape), str(valYAtt.shape)))
            print('Test:  {0:>10} {1:>12} {2:>10} {3:>20} {4:>10} {5:>12}'.format(len(testClass), str(testAtt.shape), str(testVec.shape), str(testX.shape), str(testY.shape), str(testYAtt.shape)))
        else:
            print('       {0:>10} {1:>12} {2:>10} {3:>20} {4:>10}'.format('numClass', 'classAtt', 'classVec', 'inputX', 'outputY'))
            print('Train: {0:>10} {1:>12} {2:>10} {3:>20} {4:>10}'.format(len(trainClass), str(trainAtt.shape), str(trainVec.shape), str(trainX.shape), str(trainY.shape)))
            print('Valid: {0:>10} {1:>12} {2:>10} {3:>20} {4:>10}'.format(len(valClass), str(valAtt.shape), str(valVec.shape), str(valX.shape), str(valY.shape)))
            print('Test:  {0:>10} {1:>12} {2:>10} {3:>20} {4:>10}'.format(len(testClass), str(testAtt.shape), str(testVec.shape), str(testX.shape), str(testY.shape)))

    # Attribute Modification
    tmpAtt = np.concatenate((trainAtt, valAtt, testAtt), axis=0)
    tmpWord2Vec = np.concatenate((trainVec, valVec, testVec), axis=0)
    tmpCombine = np.concatenate((tmpAtt, tmpWord2Vec), axis=1)
    if globalV.FLAGS.SELATT == 0:
        concatAtt = tmpAtt
    elif globalV.FLAGS.SELATT == 1:
        concatAtt = tmpWord2Vec
    elif globalV.FLAGS.SELATT == 2:
        concatAtt = tmpCombine


    # Check where there is some class that has same attributes
    def printClassName(pos):
        if pos < len(trainClass):
            return trainClass[pos]
        elif pos < len(trainClass) + len(valClass):
            return valClass[pos - len(trainClass)]
        else:
            return testClass[pos - len(trainClass) - len(valClass)]

    if globalV.FLAGS.HEADER == 1:
        print('\nCheck matching classes attributes')
        for i in range(concatAtt.shape[0]):
            for j in range(i + 1, concatAtt.shape[0]):
                if np.array_equal(concatAtt[i], concatAtt[j]):
                    print('{0} {1}: {2} {3}'.format(i, printClassName(i), j, printClassName(j)))
        print('')

    # Build Tree
    distanceFunc = spatial.distance.euclidean
    tmpConcatAtt = np.copy(concatAtt)
    tmpConcatName = list(trainClass + valClass + testClass)
    excludeNode = list()
    treeMap = dict()
    while len(excludeNode) != tmpConcatAtt.shape[0] - 1:
        dMatrix = np.ones((tmpConcatAtt.shape[0], tmpConcatAtt.shape[0])) * 100
        for i in range(dMatrix.shape[0]):
            for j in range(dMatrix.shape[1]):
                if i != j and i not in excludeNode and j not in excludeNode:
                    dMatrix[i][j] = distanceFunc(tmpConcatAtt[i], tmpConcatAtt[j])
        ind = np.unravel_index(np.argmin(dMatrix, axis=None), dMatrix.shape)
        tmpAvg = np.expand_dims((tmpConcatAtt[ind[0]] + tmpConcatAtt[ind[1]]) / 2, axis=0)
        tmpConcatAtt = np.concatenate((tmpConcatAtt, tmpAvg), axis=0)
        tmpConcatName.append(tmpConcatAtt.shape[0] - 1)
        excludeNode.append(ind[0])
        excludeNode.append(ind[1])
        treeMap[tmpConcatAtt.shape[0] - 1] = ind
    dot = Digraph(format='png')

    def recursiveAddEdges(index):
        if index in treeMap:
            dot.edge(str(index), str(treeMap[index][0]))
            dot.edge(str(index), str(treeMap[index][1]))
            recursiveAddEdges(treeMap[index][0])
            recursiveAddEdges(treeMap[index][1])

    for i in range(tmpConcatAtt.shape[0]):
        if i < globalV.FLAGS.numClass:
            dot.node(str(i), str(tmpConcatName[i]))
        else:
            tmpStr = distanceFunc(tmpConcatAtt[treeMap[i][0]], tmpConcatAtt[treeMap[i][1]])
            dot.node(str(i), str(round(tmpStr, 2)) + "_" + str(i))
    recursiveAddEdges(tmpConcatAtt.shape[0] - 1)
    dot.render('Tree.gv')

    # Shuffle data
    np.random.seed(globalV.FLAGS.SEED)
    s = np.arange(trainX.shape[0])
    np.random.shuffle(s)
    trainX = trainX[s]
    trainY = trainY[s]
    trainYAtt = trainYAtt[s]

    # Split train data 70/30 for each class
    trX70 = None;
    trY70 = None;
    trAtt70 = None
    trX30 = None;
    trY30 = None;
    trAtt30 = None
    for z in range(0, len(trainClass)):
        eachInputX = []
        eachInputY = []
        eachInputAtt = []
        for k in range(0, trainX.shape[0]):
            if trainY[k] == z:
                eachInputX.append(trainX[k])
                eachInputY.append(trainY[k])
                eachInputAtt.append(concatAtt[trainY[k]])
                # eachInputAtt.append(trainYAtt[k])
        eachInputX = np.array(eachInputX)
        eachInputY = np.array(eachInputY)
        eachInputAtt = np.array(eachInputAtt)
        divEach = int(eachInputX.shape[0] * 0.7)
        if trX70 is None:
            trX70 = eachInputX[:divEach]
            trY70 = eachInputY[:divEach]
            trAtt70 = eachInputAtt[:divEach]
            trX30 = eachInputX[divEach:]
            trY30 = eachInputY[divEach:]
            trAtt30 = eachInputAtt[divEach:]
        else:
            trX70 = np.concatenate((trX70, eachInputX[:divEach]), axis=0)
            trY70 = np.concatenate((trY70, eachInputY[:divEach]), axis=0)
            trAtt70 = np.concatenate((trAtt70, eachInputAtt[:divEach]), axis=0)
            trX30 = np.concatenate((trX30, eachInputX[divEach:]), axis=0)
            trY30 = np.concatenate((trY30, eachInputY[divEach:]), axis=0)
            trAtt30 = np.concatenate((trAtt30, eachInputAtt[divEach:]), axis=0)

    # Balance training class
    baX70 = None
    baAtt70 = None
    sampleEach = 500
    for z in range(len(trainClass)):
        eachInputX = []
        eachInputY = []
        for k in range(0, trX70.shape[0]):
            if trY70[k] == z:
                eachInputX.append(trX70[k])
                eachInputY.append(trAtt70[k])
        eachInputX = np.array(eachInputX)
        eachInputY = np.array(eachInputY)

        if eachInputX.shape[0] > sampleEach:
            if baX70 is None:
                baX70 = eachInputX[:sampleEach]
                baAtt70 = eachInputY[:sampleEach]
            else:
                baX70 = np.concatenate((baX70, eachInputX[:sampleEach]), axis=0)
                baAtt70 = np.concatenate((baAtt70, eachInputY[:sampleEach]), axis=0)
        else:
            duX70 = np.copy(eachInputX)
            duAtt70 = np.copy(eachInputY)
            while duX70.shape[0] < sampleEach:
                duX70 = np.concatenate((duX70, eachInputX), axis=0)
                duAtt70 = np.concatenate((duAtt70, eachInputY), axis=0)
            if baX70 is None:
                baX70 = duX70[:sampleEach]
                baAtt70 = duAtt70[:sampleEach]
            else:
                baX70 = np.concatenate((baX70, duX70[:sampleEach]), axis=0)
                baAtt70 = np.concatenate((baAtt70, duAtt70[:sampleEach]), axis=0)

    # Val class
    s = np.arange(valX.shape[0])
    tmp = list()
    for i in range(valY.shape[0]):
        tmp.append(concatAtt[valY[i] + len(trainClass)])
        # tmp.append(valYAtt[i])
    vX = valX[s]
    vY = valY[s] + len(trainClass)
    vAtt = np.array(tmp)[s]

    # Test class
    s = np.arange(testX.shape[0])
    tmp = list()
    for i in range(testY.shape[0]):
        tmp.append(concatAtt[testY[i] + len(trainClass) + len(valClass)])
        # tmp.append(testYAtt[i])
    teX = testX[s]
    teY = testY[s] + len(trainClass) + len(valClass)
    teAtt = np.array(tmp)[s]

    # Balance testing class
    baTeX = None;
    baTeY = None;
    baTeAtt = None
    sampleEach = 150
    for z in range(20, 32):
        eachInputX = []
        eachInputY = []
        eachInputAtt = []
        for k in range(0, teX.shape[0]):
            if teY[k] == z:
                eachInputX.append(teX[k])
                eachInputY.append(teY[k])
                eachInputAtt.append(teAtt[k])
        eachInputX = np.array(eachInputX)
        eachInputY = np.array(eachInputY)
        eachInputAtt = np.array(eachInputAtt)
        if baTeX is None:
            baTeX = eachInputX[:sampleEach]
            baTeY = eachInputY[:sampleEach]
            baTeAtt = eachInputAtt[:sampleEach]
        else:
            baTeX = np.concatenate((baTeX, eachInputX[:sampleEach]), axis=0)
            baTeY = np.concatenate((baTeY, eachInputY[:sampleEach]), axis=0)
            baTeAtt = np.concatenate((baTeAtt, eachInputAtt[:sampleEach]), axis=0)

    if globalV.FLAGS.HEADER == 1:
        print('Shuffle Data shape')
        print(trX70.shape, trY70.shape, trAtt70.shape)
        print(trX30.shape, trY30.shape, trAtt30.shape)
        print(vX.shape, vY.shape, vAtt.shape)
        print(teX.shape, teY.shape, teAtt.shape)
        print(baX70.shape, baAtt70.shape)

    returnData = dataClass()
    returnData.trX70 = trX70
    returnData.trY70 = trY70
    returnData.trAtt70 = trAtt70
    returnData.trX30 = trX30
    returnData.trY30 = trY30
    returnData.trAtt30 = trAtt30
    returnData.vX = vX
    returnData.vY = vY
    returnData.vAtt = vAtt
    returnData.teX = teX
    returnData.teY = teY
    returnData.teAtt = teAtt
    returnData.baX70 = baX70
    returnData.baAtt70 = baAtt70
    returnData.concatAtt = concatAtt
    returnData.allClassName = trainClass + valClass + testClass
    returnData.treeMap = treeMap
    returnData.distanceFunc = distanceFunc
    returnData.tmpConcatAtt = tmpConcatAtt

    return returnData

def accuracy(attObj, classifyObj, x, y):
    """
    Calculate accuracy based on attribute and classify model.
    :param attObj:
    :param classifyObj:
    :param x:
    :param y:
    :return: accuracy in Percentage
    """
    tmp1 = attObj.getAttribute(x)
    tmp2 = classifyObj.predict(tmp1)
    return np.mean(np.equal(tmp2, y))

def euclidean (attObj, x, y, concatAttD):
    """
    Calculate accuracy based on attribute and classify model.
    :param attObj:
    :param x:
    :param y:
    :param concatAttD:
    :return: accuracy in Percentage
    """
    euTmp = attObj.getAttribute(x)
    tmp2 = []
    for pp in euTmp:
        tmpDistance = []
        for cc in concatAttD:
            tmpDistance.append(spatial.distance.euclidean(pp, cc))
        tmpIndex = np.argsort(tmpDistance)[:1]
        tmp2.append(tmpIndex[0])
    return np.mean(np.equal(tmp2, y))

def topAccuracy(attObj, classifyObj, x, y, top):
    """
    Calculate top accuracy.
    :param attObj:
    :param classifyObj:
    :param x:
    :param y:
    :param top:
    :return:
    """
    tmpAtt = attObj.getAttribute(x)
    tmpScore = classifyObj.predictScore(tmpAtt)
    tmpSort = np.argsort(-tmpScore, axis=1)
    tmpPred = tmpSort[:, :top]
    count = 0
    for i, p in enumerate(tmpPred):
        if y[i] in p:
            count += 1
    return count / x.shape[0]

if __name__ == "__main__":
    argumentParser()

    createFolder()

    dataSet = loadDataset()

    if globalV.FLAGS.OPT == 1:
        print('\nTrain Attribute')
        attModel = attribute()
        attModel.trainAtt(dataSet.baX70, dataSet.baAtt70, dataSet.vX, dataSet.vAtt, dataSet.teX, dataSet.teAtt)

    elif globalV.FLAGS.OPT == 2:
        print('\nTrain Classify')
        tmpAttributes = dataSet.concatAtt.copy()
        tmpClassIndex = np.arange(dataSet.concatAtt.shape[0])
        numberOfSample = 35
        for i in range (globalV.FLAGS.numClass):
            if i < 12:
                countSample = 0
                for j in range(dataSet.trX30.shape[0]):
                    if dataSet.trY30[j] == i:
                        tmpAttributes = np.concatenate((tmpAttributes, np.expand_dims(dataSet.trAtt30[j], axis=0)), axis=0)
                        tmpClassIndex = np.concatenate((tmpClassIndex, np.expand_dims(i, axis=0)), axis=0)
                        countSample += 1
                    if countSample == numberOfSample:
                        break
            else:
                for j in range(numberOfSample):
                    tmpAttributes = np.concatenate((tmpAttributes, np.expand_dims(dataSet.concatAtt[i], axis=0)), axis=0)
                    tmpClassIndex = np.concatenate((tmpClassIndex, np.expand_dims(i, axis=0)), axis=0)
        classifier = classify()
        # classifier.trainClassify(concatAtt_D, np.arange(dataSet.concatAtt.shape[0]), 0.5)
        classifier.trainClassify(tmpAttributes, tmpClassIndex, 0.5)

    elif globalV.FLAGS.OPT == 3:
        print('\nAccuracy')
        g1 = tf.Graph()
        g2 = tf.Graph()
        with g1.as_default():
            model = attribute()
        with g2.as_default():
            classifier = classify()

        if not os.path.isfile('Report_OPT_3.csv'):
            with open("Report_OPT_3.csv", "a") as file:
                file.write('Key, TrAcc, VAcc, TeAcc, EuTAcc, EuVAcc, EuTeAcc, Top1Tr, Top1V, Top1Te, Top1Avg,'
                           'Top3Tr, Top3V, Top3Te, Top3Avg,'
                           'Top5Tr, Top5V, Top5Te, Top5Avg,'
                           'Top7Tr, Top7V, Top7Te, Top7Avg,'
                           'Top10Tr, Top10V, Top10Te, Top10Avg,'
                           'person, statue, car, areoplane, cat, zebra, sheep, bicycle, bottle, sofa, carriage, bird, potted_plant,'
                           'tv_monitor, building, centaur, train, donkey, jetski, dining_table, monkey, bus, wolf, dog, horse,'
                           'cow, motorbike, mug, chair, boat, bag, goat\n')

        myFile = open("Report_OPT_3.csv", "a")
        myFile.write('{0}'.format(globalV.FLAGS.DIR))

        # Classify accuracy
        tmpAcc = accuracy(model, classifier, dataSet.trX30, dataSet.trY30) * 100
        print('Train accuracy = {0:.4f}%'.format(tmpAcc))
        myFile.write(',{0}'.format(tmpAcc))
        tmpAcc = accuracy(model, classifier, dataSet.vX, dataSet.vY) * 100
        print('Val accuracy = {0:.4f}%'.format(tmpAcc))
        myFile.write(',{0}'.format(tmpAcc))
        tmpAcc = accuracy(model, classifier, dataSet.teX, dataSet.teY) * 100
        print('Test accuracy = {0:.4f}%'.format(tmpAcc))
        myFile.write(',{0}'.format(tmpAcc))

        # Euclidean accuracy
        tmpAcc = euclidean(model, dataSet.trX30, dataSet.trY30, dataSet.concatAtt) * 100
        print('Euclidean train accuracy = {0:.4f}%'.format(tmpAcc))
        myFile.write(',{0}'.format(tmpAcc))
        tmpAcc = euclidean(model, dataSet.vX, dataSet.vY, dataSet.concatAtt) * 100
        print('Euclidean val accuracy = {0:.4f}%'.format(tmpAcc))
        myFile.write(',{0}'.format(tmpAcc))
        tmpAcc = euclidean(model, dataSet.teX, dataSet.teY, dataSet.concatAtt) * 100
        print('Euclidean test accuracy = {0:.4f}%'.format(tmpAcc))
        myFile.write(',{0}'.format(tmpAcc))


        # Top Accuracy
        topAccList = [1, 3, 5, 7, 10]
        for topAcc in topAccList:
            tmpAcc = topAccuracy(model, classifier, dataSet.trX30, dataSet.trY30, topAcc) * 100
            print('Top {0} train Accuracy = {1:.4f}%'.format(topAcc, tmpAcc))
            myFile.write(',{0}'.format(tmpAcc))
            tmpAcc = topAccuracy(model, classifier, dataSet.vX, dataSet.vY, topAcc) * 100
            print('Top {0} val Accuracy = {1:.4f}%'.format(topAcc, tmpAcc))
            myFile.write(',{0}'.format(tmpAcc))
            tmpAcc = topAccuracy(model, classifier, dataSet.teX, dataSet.teY, topAcc) * 100
            print('Top {0} test Accuracy = {1:.4f}%'.format(topAcc, tmpAcc))
            myFile.write(',{0}'.format(tmpAcc))
            tmpAcc = (((topAccuracy(model, classifier, dataSet.teX, dataSet.teY, topAcc) * dataSet.trX30.shape[0]) +
                       (topAccuracy(model, classifier, dataSet.vX, dataSet.vY, topAcc) * dataSet.vX.shape[0]) +
                       (topAccuracy(model, classifier, dataSet.teX, dataSet.teY, topAcc) * dataSet.teX.shape[0])) /
                      (dataSet.trX30.shape[0] + dataSet.vX.shape[0] + dataSet.teX.shape[0])) * 100
            print('Top {0} average Accuracy = {1:.4f}%'.format(topAcc, tmpAcc))
            myFile.write(',{0}'.format(tmpAcc))

        # Accuracy each class
        def calConfusion(inputX):
            attTmp1 = model.getAttribute(inputX)
            tmpScore1 = classifier.predictScore(attTmp1)
            tmpSort1 = np.argsort(-tmpScore1, axis=1)
            tmpPred1 = tmpSort1[:, :1]
            tmpPred1 = np.reshape(tmpPred1, -1)
            tmpCountEach1 = np.bincount(tmpPred1, minlength=globalV.FLAGS.numClass)
            tmpCountEach1 = np.array(tmpCountEach1) / (inputX.shape[0] * 1)
            return tmpCountEach1

        def accEachClass(start, end, x1, y1, className, confusionRef):
            for z in range(start, end):
                eachInputX = []
                eachInputY = []
                for k in range(0, x1.shape[0]):
                    if y1[k] == z:
                        eachInputX.append(x1[k])
                        eachInputY.append(y1[k])
                eachInputX = np.array(eachInputX)
                eachInputY = np.array(eachInputY)
                tmpAcc = accuracy(model, classifier, eachInputX, eachInputY) * 100
                print('Class: {0:<15} Size: {1:<10} Accuracy = {2:.4f}%'.format(className[z], eachInputX.shape[0], tmpAcc))
                myFile.write(',{0}'.format(tmpAcc))
                confusion.append(calConfusion(eachInputX))

        confusion = []
        accEachClass(0, 15, dataSet.trX30, dataSet.trY30, dataSet.allClassName, confusion)
        accEachClass(15, 20, dataSet.vX, dataSet.vY, dataSet.allClassName, confusion)
        accEachClass(20, 32, dataSet.teX, dataSet.teY, dataSet.allClassName, confusion)

        # Show confusion matrix
        confusion = np.array(confusion)
        tmpClassName = [dataSet.allClassName[x] for x in range(globalV.FLAGS.numClass)]
        trace = go.Heatmap(z=confusion,
                           x=tmpClassName,
                           y=tmpClassName,
                           zmax=1.0,
                           zmin=0.0
                           )
        data = [trace]
        layout = go.Layout(title=globalV.FLAGS.KEY, width=1920, height=1080,
                           yaxis=dict(
                               ticktext=tmpClassName,
                               tickvals=np.arange(len(tmpClassName))))
        fig = go.Figure(data=data, layout=layout)
        py.image.save_as(fig, filename=globalV.FLAGS.DIR + '_Confusion_.png')

        # Show classes Heat map
        with open(globalV.FLAGS.BASEDIR + globalV.FLAGS.APYPATH + 'attribute_names.txt', 'r') as f:
            allClassAttName = [line.strip() for line in f]
        trace = go.Heatmap(z=dataSet.concatAtt,
                           x=allClassAttName,
                           y=dataSet.allClassName,
                           zmax=1.0,
                           zmin=0.0
                           )
        data = [trace]
        layout = go.Layout(title=globalV.FLAGS.KEY, width=1920, height=1080)
        fig = go.Figure(data=data, layout=layout)
        py.image.save_as(fig, filename=globalV.FLAGS.DIR+'_Heat.png')

    elif globalV.FLAGS.OPT == 4:
        print('\nPredict classes from cluster')

        mapAll = [1, 1, 2, 2, 0, 0, 0, 3, 4, 6, 5, 7, 8, 9, 5, 1, 2, 0, 3, 5, 1, 2, 0, 0, 0, 0, 3, 4, 6, 5, 5, 0]

        # Train Class Map
        trYCluster70 = np.copy(dataSet.trY70)
        trYCluster30 = np.copy(dataSet.trY30)
        for i in range(dataSet.trY70.shape[0]):
            trYCluster70[i] = mapAll[dataSet.trY70[i]]
        for i in range(dataSet.trY30.shape[0]):
            trYCluster30[i] = mapAll[dataSet.trY30[i]]

        # Validation Class Map
        vYCluster = np.copy(dataSet.vY)
        for i in range(dataSet.vY.shape[0]):
            vYCluster[i] = mapAll[dataSet.vY[i]]

        # Test Class Map
        teYCluster = np.copy(dataSet.teY)
        for i in range(dataSet.teY.shape[0]):
            teYCluster[i] = mapAll[dataSet.teY[i]]


        # Find cluster with Graph
        returnIndex = [57, 11, 53, 12, 40, 51, 13, 43, 47, 52]
        returnValue = [1, 7, 0, 8, 6, 3, 9, 4, 5, 2]

        def FindCluster(att, pos):
            if pos in returnIndex:
                return returnValue[returnIndex.index(pos)]
            else:
                left = dataSet.distanceFunc(dataSet.tmpConcatAtt[dataSet.treeMap[pos][0]], att)
                right = dataSet.distanceFunc(dataSet.tmpConcatAtt[dataSet.treeMap[pos][1]], att)
                if left < right:
                    return FindCluster(att, dataSet.treeMap[pos][0])
                else:
                    return FindCluster(att, dataSet.treeMap[pos][1])

        g1 = tf.Graph()
        g2 = tf.Graph()
        with g1.as_default():
            model = attribute()
        with g2.as_default():
            classifier = classify()

        def clusterClassAcc(x, yCluster, y, keyword):
            tmpAtt_0 = model.getAttribute(x)
            tmpScore_0 = classifier.predictScore(tmpAtt_0)
            tmpSort_0 = np.argsort(-tmpScore_0, axis=1)
            tmpPredCluster_0 = []
            for i_0 in range(tmpAtt_0.shape[0]):
                tmpPredCluster_0.append(FindCluster(tmpAtt_0[i_0], dataSet.tmpConcatAtt.shape[0] - 1))
            tmpPredCluster_0 = np.array(tmpPredCluster_0)
            countCorrect_0 = 0
            for i_0 in range(tmpPredCluster_0.shape[0]):
                if tmpPredCluster_0[i_0] == yCluster[i_0]:
                    countTop_0 = 0
                    for j in range(tmpSort_0[i_0].shape[0]):
                        if mapAll[tmpSort_0[i_0][j]] == yCluster[i_0]:
                            countTop_0 += 1
                            if tmpSort_0[i_0][j] == y[i_0]:
                                countCorrect_0 += 1
                                break
                            if countTop_0 == 1:
                                break

            predY_0 = classifier.predict(tmpAtt_0)
            print('{0} Accuracy = {1:.4f}%'.format(keyword, np.mean(np.equal(predY_0, y)) * 100))
            print('Cluster {0} Accuracy = {1:.4f}%'.format(keyword, np.mean(np.equal(tmpPredCluster_0, yCluster)) * 100))
            print('Cluster+Classes {0} Accuracy = {1:.4f}%'.format(keyword, (countCorrect_0 / tmpPredCluster_0.shape[0]) * 100))
            countCorrect_0 = 0
            for i_0 in range(predY_0.shape[0]):
                if mapAll[predY_0[i_0]] == mapAll[y[i_0]]:
                    countCorrect_0 += 1
            print('Check predY has same cluster {0} Accuracy = {1:.4f}%'.format(keyword, (countCorrect_0 / predY_0.shape[0]) * 100))

        clusterClassAcc(dataSet.trX30, trYCluster30, dataSet.trY30, "Train")
        clusterClassAcc(dataSet.vX, vYCluster, dataSet.vY, "Val")
        clusterClassAcc(dataSet.teX, teYCluster, dataSet.teY, "Val")



