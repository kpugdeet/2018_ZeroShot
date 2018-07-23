import numpy as np
import tensorflow as tf
import argparse
import os
import plotly.plotly as py
import plotly.graph_objs as go
from scipy import spatial
from graphviz import Digraph
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

"""
import created files
"""
import globalV
from loadData import loadData
from attribute import attribute
from classify import classify
from softmax import softmax

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
    lenTr = None
    originalAtt = None

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
    parser.add_argument('--DIR', type=str, default='APY_249', help='Choose working directory')
    parser.add_argument('--numClass', type=int, default=32, help='Number of class')

    # Hyper Parameter
    parser.add_argument('--maxSteps', type=int, default=100, help='Number of steps to run trainer.')
    parser.add_argument('--lr', type=float, default=1e-7, help='Initial learning rate')
    parser.add_argument('--batchSize', type=int, default=32, help='Batch size')

    # Initialize or Restore Model
    parser.add_argument('--TD', type=int, default=0, help='0.Restore 1.Train visual model')
    parser.add_argument('--TA', type=int, default=0, help='0.Restore 1.Train attribute model')
    parser.add_argument('--TC', type=int, default=0, help='0.Restore 1.Train classify model')

    # Choose what to do
    parser.add_argument('--OPT', type=int, default=0, help='0.None, 1.Attribute, 2.Classify, 3.Accuracy')

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

    if globalV.FLAGS.KEY == "APY":
        with open(globalV.FLAGS.BASEDIR + globalV.FLAGS.APYPATH + 'attribute_names.txt', 'r') as f:
            allClassAttName = [line.strip() for line in f]

    # Based on TreeMap
    tmpReduceAtt = np.zeros((tmpAtt.shape[0], 31))
    for k in range(tmpAtt.shape[0]):
        if tmpAtt[k][allClassAttName.index('Wing')] > 0.108:
            tmpReduceAtt[k][0] = 1

        if tmpAtt[k][allClassAttName.index('Ear')] > 0.267:
            tmpReduceAtt[k][1] = 1
        if tmpAtt[k][allClassAttName.index('Metal')] > 0.472:
            tmpReduceAtt[k][2] = 1

        if tmpAtt[k][allClassAttName.index('Text')] > 0.016:
            tmpReduceAtt[k][3] = 1
        if tmpAtt[k][allClassAttName.index('Hand')] > 0.272:
            tmpReduceAtt[k][4] = 1
        if tmpAtt[k][allClassAttName.index('Tail')] > 0.709:
            tmpReduceAtt[k][5] = 1

        if tmpAtt[k][allClassAttName.index('Stem/Trunk')] > 0.253:
            tmpReduceAtt[k][6] = 1
        if tmpAtt[k][allClassAttName.index('Wood')] > 0.17:
            tmpReduceAtt[k][7] = 1
        if tmpAtt[k][allClassAttName.index('Ear')] > 0.645:
            tmpReduceAtt[k][8] = 1
        if tmpAtt[k][allClassAttName.index('Glass')] > 0.002:
            tmpReduceAtt[k][9] = 1

        if tmpAtt[k][allClassAttName.index('Screen')] > 0.421:
            tmpReduceAtt[k][10] = 1
        if tmpAtt[k][allClassAttName.index('Cloth')] > 0.176:
            tmpReduceAtt[k][11] = 1
        if tmpAtt[k][allClassAttName.index('2D Boxy')] > 0.11:
            tmpReduceAtt[k][12] = 1
        if tmpAtt[k][allClassAttName.index('Tail')] > 0.354:
            tmpReduceAtt[k][13] = 1
        if tmpAtt[k][allClassAttName.index('Foot/Shoe')] > 0.424:
            tmpReduceAtt[k][14] = 1

        if tmpAtt[k][allClassAttName.index('Furn. Seat')] > 0.283:
            tmpReduceAtt[k][15] = 1
        if tmpAtt[k][allClassAttName.index('Handlebars')] > 0.78:
            tmpReduceAtt[k][16] = 1
        if tmpAtt[k][allClassAttName.index('Exhaust')] > 0.01:
            tmpReduceAtt[k][17] = 1
        if tmpAtt[k][allClassAttName.index('Saddle')] > 0.04:
            tmpReduceAtt[k][18] = 1

        if tmpAtt[k][allClassAttName.index('Clear')] > 0.025:
            tmpReduceAtt[k][19] = 1
        if tmpAtt[k][allClassAttName.index('Furn. Seat')] > 0.678:
            tmpReduceAtt[k][20] = 1
        if tmpAtt[k][allClassAttName.index('Headlight')] > 0.283:
            tmpReduceAtt[k][21] = 1
        if tmpAtt[k][allClassAttName.index('Snout')] > 0.864:
            tmpReduceAtt[k][22] = 1
        if tmpAtt[k][allClassAttName.index('Ear')] > 0.884:
            tmpReduceAtt[k][23] = 1

        if tmpAtt[k][allClassAttName.index('Taillight')] > 0.101:
            tmpReduceAtt[k][24] = 1
        if tmpAtt[k][allClassAttName.index('Round')] > 0.163:
            tmpReduceAtt[k][25] = 1
        if tmpAtt[k][allClassAttName.index('Tail')] > 0.463:
            tmpReduceAtt[k][26] = 1
        if tmpAtt[k][allClassAttName.index('Occluded')] > 0.334:
            tmpReduceAtt[k][27] = 1

        if tmpAtt[k][allClassAttName.index('Exhaust')] > 0.069:
            tmpReduceAtt[k][28] = 1
        if tmpAtt[k][allClassAttName.index('Metal')] > 0.074:
            tmpReduceAtt[k][29] = 1
        if tmpAtt[k][allClassAttName.index('Saddle')] > 0.243:
            tmpReduceAtt[k][30] = 1


    if globalV.FLAGS.SELATT == 0:
        concatAtt = tmpAtt
    elif globalV.FLAGS.SELATT == 1:
        concatAtt = tmpWord2Vec
    elif globalV.FLAGS.SELATT == 2:
        concatAtt = tmpCombine
    elif globalV.FLAGS.SELATT == 3:
        concatAtt = tmpReduceAtt


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
    # trainYAtt = trainYAtt[s]

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
    returnData.lenTr = len(trainClass)
    returnData.originalAtt = tmpAtt

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

    if globalV.FLAGS.KEY == "APY":
        with open(globalV.FLAGS.BASEDIR + globalV.FLAGS.APYPATH + 'attribute_names.txt', 'r') as f:
            allClassAttName = [line.strip() for line in f]
    elif globalV.FLAGS.KEY == "AWA2":
        with open(globalV.FLAGS.BASEDIR + globalV.FLAGS.AWA2PATH + 'predicates.txt', 'r') as f:
            allClassAttName = [line.split()[1] for line in f]

    # from sklearn import tree
    # clf = tree.DecisionTreeClassifier()
    # clf = clf.fit(dataSet.concatAtt, range(len(dataSet.allClassName)))
    #
    # import graphviz
    # dot_data = tree.export_graphviz(clf, out_file=None,
    #                                 feature_names=allClassAttName,
    #                                 class_names=dataSet.allClassName,
    #                                 filled=True, rounded=True,
    #                                 special_characters=True)
    # graph = graphviz.Source(dot_data)
    # graph.render(globalV.FLAGS.KEY)
    #
    #
    # trace = go.Heatmap(z=dataSet.concatAtt,
    #                    x=allClassAttName,
    #                    y=dataSet.allClassName,
    #                    zmax=1.0,
    #                    zmin=0.0
    #                    )
    # data = [trace]
    # layout = go.Layout(title=globalV.FLAGS.KEY, width=1920, height=1080)
    # fig = go.Figure(data=data, layout=layout)
    # py.image.save_as(fig, filename=globalV.FLAGS.DIR + '_Heat.png')

    # for n_clusters in range(2, 32):
    #     kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(dataSet.concatAtt)
    #     silhouette_avg = silhouette_score(dataSet.concatAtt, kmeans.labels_)
    #     print('For n_clusters = {0} The average silhouette_score is : {1}'.format(n_clusters, silhouette_avg))


    # kmeanPred = KMeans(n_clusters=10, random_state=0).fit(dataSet.concatAtt)
    # mapAll = []
    # for i in range(dataSet.concatAtt.shape[0]):
    #     print('{0}::{1}'.format(dataSet.allClassName[i],kmeanPred.labels_[i]))
    #     mapAll.append(kmeanPred.labels_[i])
    # mapAll = np.array(mapAll)

    if globalV.FLAGS.OPT == 1:
        print('\nTrain Attribute')
        attModel = attribute()
        attModel.trainAtt(dataSet.trX70, dataSet.trAtt70, dataSet.vX, dataSet.vAtt, dataSet.teX, dataSet.teAtt)

    elif globalV.FLAGS.OPT == 2:
        print('\nTrain Classify')
        tmpAttributes = dataSet.concatAtt.copy()
        tmpClassIndex = np.arange(dataSet.concatAtt.shape[0])
        numberOfSample = 35
        for i in range (globalV.FLAGS.numClass):
            if i < dataSet.lenTr:
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

        if globalV.FLAGS.KEY == 'APY':
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

        myFile = open("Report_OPT_3_"+globalV.FLAGS.KEY+".csv", "a")
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
        myFile.write('\n'.format(tmpAcc))
        myFile.close()

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
        if globalV.FLAGS.KEY == "APY":
            with open(globalV.FLAGS.BASEDIR + globalV.FLAGS.APYPATH + 'attribute_names.txt', 'r') as f:
                allClassAttName = [line.strip() for line in f]
        elif globalV.FLAGS.KEY == "AWA2":
            with open(globalV.FLAGS.BASEDIR + globalV.FLAGS.AWA2PATH + 'predicates.txt', 'r') as f:
                allClassAttName = [line.split()[1] for line in f]

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

    elif globalV.FLAGS.OPT == 5:
        print('\nTrain Binary')

        tmp_combineX = np.concatenate((dataSet.trX70, dataSet.trX30, dataSet.vX, dataSet.teX), axis=0)
        tmp_combineY = np.concatenate((dataSet.trY70, dataSet.trY30, dataSet.vY, dataSet.teY), axis=0)
        s = np.arange(tmp_combineX.shape[0])
        np.random.shuffle(s)
        tmp_combineX = tmp_combineX[s]
        tmp_combineY = tmp_combineY[s]
        print(tmp_combineX.shape, tmp_combineY.shape)

        # Balance each class Data
        pickNum = 300
        combineX = []
        combineY = []
        for i in range(len(dataSet.allClassName)):
            count = 0
            for j in range(tmp_combineY.shape[0]):
                if tmp_combineY[j] == i:
                    combineX.append(tmp_combineX[j])
                    combineY.append(tmp_combineY[j])
                    count += 1
                if count == pickNum:
                    break
        combineX = np.array(combineX)
        combineY = np.array(combineY)
        print(combineX.shape, combineY.shape)


        if globalV.FLAGS.KEY == "APY":
            with open(globalV.FLAGS.BASEDIR + globalV.FLAGS.APYPATH + 'attribute_names.txt', 'r') as f:
                allClassAttName = [line.strip() for line in f]

        """
        Wing Nodes (lr=1e-7)
        """
        # indexAtt = allClassAttName.index('Wing')
        # _trX = []
        # _trY = []
        # _teX = []
        # _teY = []
        # for i in range(combineX.shape[0]):
        #     if dataSet.allClassName[combineY[i]] not in ['person']:
        #         if dataSet.concatAtt[combineY[i]][indexAtt] <= 0.108:
        #             if dataSet.allClassName[combineY[i]] in ['car', 'mug', 'sofa', 'bicycle', 'boat', 'zebra', 'wolf', 'cow', 'goat']:
        #                 _teX.append(combineX[i])
        #                 _teY.append(0)
        #             else:
        #                 _trX.append(combineX[i])
        #                 _trY.append(0)
        #         else:
        #             if i%2 == 0:
        #                 _trX.append(combineX[i])
        #                 _trY.append(1)
        #             else:
        #                 _teX.append(combineX[i])
        #                 _teY.append(1)
        # _trX = np.array(_trX)
        # _trY = np.array(_trY)
        # _teX = np.array(_teX)
        # _teY = np.array(_teY)
        # print(_trX.shape, _trY.shape)
        # print(_teX.shape, _teY.shape)
        # print(np.sum(_trY), np.sum(_teY))

        """
        Ear Nodes (lr=1e-7)
        """
        # #
        # indexAtt = allClassAttName.index('Ear')
        # _trX = []
        # _trY = []
        # _teX = []
        # _teY = []
        # for i in range(combineX.shape[0]):
        #     if dataSet.allClassName[combineY[i]] not in ['bird', 'Aeroplane', 'centaur', 'person']:
        #         if dataSet.concatAtt[combineY[i]][indexAtt] <= 0.267:
        #             if dataSet.allClassName[combineY[i]] in ['car', 'mug', 'sofa', 'bicycle', 'boat']:
        #                 _teX.append(combineX[i])
        #                 _teY.append(0)
        #             else:
        #                 _trX.append(combineX[i])
        #                 _trY.append(0)
        #         else:
        #             if dataSet.allClassName[combineY[i]] in ['zebra', 'wolf', 'cow', 'goat']:
        #                 _teX.append(combineX[i])
        #                 _teY.append(1)
        #             else:
        #                 _trX.append(combineX[i])
        #                 _trY.append(1)
        # _trX = np.array(_trX)
        # _trY = np.array(_trY)
        # _teX = np.array(_teX)
        # _teY = np.array(_teY)
        # print(_trX.shape, _trY.shape)
        # print(_teX.shape, _teY.shape)

        """
        Hand Nodes (lr=1e-7)
        """
        _trX = []
        _trY = []
        _teX = []
        _teY = []
        for i in range(combineX.shape[0]):
            if dataSet.concatAtt[combineY[i]][allClassAttName.index('Wing')] <= 0.108 and dataSet.concatAtt[combineY[i]][allClassAttName.index('Ear')] > 0.267:
                if dataSet.concatAtt[combineY[i]][allClassAttName.index('Hand')] <= 0.272:
                    if dataSet.allClassName[combineY[i]] in ['zebra', 'wolf', 'cow', 'goat']:
                        # _teX.append(combineX[i])
                        # _teY.append(0)
                        None
                    else:
                        _trX.append(combineX[i])
                        _trY.append(0)
                elif dataSet.allClassName[combineY[i]] not in ['monkey']:
                    if dataSet.allClassName[combineY[i]] in ['statue']:
                        _teX.append(combineX[i])
                        _teY.append(1)
                        # None
                    else:
                        _trX.append(combineX[i])
                        _trY.append(1)
        _trX = np.array(_trX)
        _trY = np.array(_trY)
        _teX = np.array(_teX)
        _teY = np.array(_teY)
        print(_trX.shape, _trY.shape)
        print(_teX.shape, _teY.shape)
        softModel = softmax()
        softModel.train(_trX, _trY, _teX, _teY)

    elif globalV.FLAGS.OPT == 6:
        from sklearn import tree
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(dataSet.originalAtt, range(len(dataSet.allClassName)))


        g1 = tf.Graph()
        g2 = tf.Graph()
        with g1.as_default():
            model = attribute()
        with g2.as_default():
            classifier = classify()


        def transformBack (reduceAtt):
            tmpFullAtt = np.zeros((reduceAtt.shape[0], 64))
            indexMapping = [21, 9, 52, 37, 17, 6, 48, 54, 9, 57, 50, 55, 0, 6, 20, 41, 33, 31, 45, 60, 41, 28, 10, 9, 29, 2, 6, 5, 31, 52, 45]
            for k in range(reduceAtt.shape[0]):
                for l in range(reduceAtt.shape[1]):
                    tmpFullAtt[k][indexMapping[l]] = reduceAtt[k][l]
            return tmpFullAtt


        def clusterClassAcc(start, end, x1, y1, keyword):

            for z in range(start, end):
                eachInputX = []
                eachInputY = []
                for k in range(0, x1.shape[0]):
                    if y1[k] == z:
                        eachInputX.append(x1[k])
                        eachInputY.append(y1[k])
                eachInputX = np.array(eachInputX)
                eachInputY = np.array(eachInputY)

                tmpAtt_1 = model.getAttribute(eachInputX)

                # predY_1 = classifier.predict(tmpAtt_1)
                tmpAtt_1 = transformBack(tmpAtt_1)
                tmpAtt_1 = np.where(tmpAtt_1 > 0.5, 1, 0)
                predY_1 = clf.predict(tmpAtt_1)

                tmpAcc_1 = np.mean(np.equal(predY_1, eachInputY))*100
                print('Class: {0:<15} Size: {1:<10} Accuracy = {2:.4f}%'.format(dataSet.allClassName[z], eachInputX.shape[0], tmpAcc_1))


            # tmpAtt_0 = model.getAttribute(x)
            # predY_1 = classifier.predict(tmpAtt_0)
            #
            # tmpAtt_0 = transformBack(tmpAtt_0)
            # tmpAtt_0 = np.where(tmpAtt_0 > 0.5, 1, 0)
            #
            #
            # selectClass = 2
            # showAtt = []
            # showAtt.append(dataSet.concatAtt[selectClass])
            # print(dataSet.allClassName[selectClass])
            # for m in range(y.shape[0]):
            #     if y[m] == selectClass:
            #         showAtt.append(tmpAtt_0[m])
            #
            # # showAtt = np.array(showAtt)
            # # trace = go.Heatmap(z=showAtt,
            # #                    x=allClassAttName,
            # #                    zmax=1.0,
            # #                    zmin=0.0
            # #                    )
            # # data = [trace]
            # # layout = go.Layout(title=globalV.FLAGS.KEY, width=1920, height=1080)
            # # fig = go.Figure(data=data, layout=layout)
            # # py.image.save_as(fig, filename=globalV.FLAGS.DIR + '_Test.png')
            #
            #
            # predY_0 = clf.predict(tmpAtt_0)
            # print('{0} Accuracy = {1:.4f}%'.format(keyword, np.mean(np.equal(predY_1, y)) * 100))
            # print('{0} Accuracy = {1:.4f}%'.format(keyword, np.mean(np.equal(predY_0, y)) * 100))

        clusterClassAcc(0, 15, dataSet.trX70, dataSet.trY70, "Train70")
        # clusterClassAcc(dataSet.trX30, dataSet.trY30, "Train30")
        # clusterClassAcc(dataSet.vX, dataSet.vY, "Val")
        # clusterClassAcc(dataSet.teX, dataSet.teY, "Test")

        # def accEachClass(start, end, x1, y1, className):
        #     for z in range(start, end):
        #         eachInputX = []
        #         eachInputY = []
        #         for k in range(0, x1.shape[0]):
        #             if y1[k] == z:
        #                 eachInputX.append(x1[k])
        #                 eachInputY.append(y1[k])
        #         eachInputX = np.array(eachInputX)
        #         eachInputY = np.array(eachInputY)
        #         tmpAcc_1 = accuracy(model, classifier, eachInputX, eachInputY) * 100
        #         print('Class: {0:<15} Size: {1:<10} Accuracy = {2:.4f}%'.format(className[z], eachInputX.shape[0], tmpAcc_1))
        # accEachClass(0, 15, dataSet.trX30, dataSet.trY30, dataSet.allClassName)
        # accEachClass(15, 20, dataSet.vX, dataSet.vY, dataSet.allClassName)
        # accEachClass(20, 32, dataSet.teX, dataSet.teY, dataSet.allClassName)





