import numpy as np
import argparse
import os
from sklearn.neighbors import NearestNeighbors

"""
import created files
"""
from loadData import loadData
from attribute import attribute

"""
Global Variable
"""
import globalV

class dataClass:
    """
    Data class to keep all data to train and test model
    """
    concatAtt = None
    concatAttName = None
    allClassName = None
    trX70 = None
    trY70 = None
    trAtt70 = None
    trX30 = None
    trY30 = None
    trAtt30 = None
    baX70 = None
    baY70 = None
    baAtt70 = None
    vX = None
    vY = None
    vAtt = None
    teX = None
    teY = None
    teAtt = None

def argumentParser():
    """
    Get all argument values from command line.
    :return: No return, set value to globalV variable.
    """
    parser = argparse.ArgumentParser()

    # Input image size & attribute dimension
    parser.add_argument('--width', type=int, default=227, help='Image width')
    parser.add_argument('--height', type=int, default=227, help='Image height')
    parser.add_argument('--SELATT', type=int, default=0, help='0.Att, 1.Word2Vec, 2.Att+Word2Vec, 3.Reduced Att')
    parser.add_argument('--numAtt', type=int, default=64, help='Dimension of Attribute')
    parser.add_argument('--numClass', type=int, default=15, help='Number of class')

    # Dataset Path
    parser.add_argument('--BASEDIR', type=str, default='/media/dataHD3/kpugdeet/', help='Base folder for dataset and logs')
    parser.add_argument('--AWA2PATH', type=str, default='AWA2/Animals_with_Attributes2/', help='AWA2 dataset')
    parser.add_argument('--CUBPATH', type=str, default='CUB/CUB_200_2011/', help='CUB dataset')
    parser.add_argument('--SUNPATH', type=str, default='SUN/SUNAttributeDB/', help='SUN dataset')
    parser.add_argument('--APYPATH', type=str, default='APY/attribute_data/', help='APY dataset')
    parser.add_argument('--GOOGLE', type=str, default='PRE/GoogleNews-vectors-negative300.bin', help='Google Word2Vec model')

    # Working directory
    parser.add_argument('--KEY', type=str, default='APY', help='Choose dataset (AWA2, CUB, SUN, APY)')
    parser.add_argument('--DIR', type=str, default='APY_2', help='Choose working directory')

    # Hyper Parameter
    parser.add_argument('--maxSteps', type=int, default=100, help='Number of steps to run trainer.')
    parser.add_argument('--lr', type=float, default=1e-7, help='Initial learning rate')
    parser.add_argument('--batchSize', type=int, default=32, help='Batch size')

    # Initialize or Restore Model
    parser.add_argument('--TA', type=int, default=0, help='0.Restore 1.Train')

    # Choose what to do
    parser.add_argument('--OPT', type=int, default=2, help='0.None')

    # ETC.
    parser.add_argument('--HEADER', type=int, default=1, help='0.Not-Show, 1.Show')
    parser.add_argument('--SEED', type=int, default=0, help='Random number for shuffle data')

    # Model
    parser.add_argument('--MODEL', type=str, default='/attribute/model/model.ckpt', help='Random number for shuffle data')

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

def loadDataset():
    """
    Load data set and split to Train, Validatio, and Test set.
    Choose whether to use attribute, word2Vec, or attribute+word2Vec to train attribute and classify model.

    :return:
    """
    (trainClass, trainAtt, trainVec, trainX, trainY, trainYAtt), \
    (valClass, valAtt, valVec, valX, valY, valYAtt), \
    (testClass, testAtt, testVec, testX, testY, testYAtt) = loadData.getData()
    returnData = dataClass()

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
    if globalV.FLAGS.SELATT == 0:
        returnData.concatAtt = np.concatenate((trainAtt, valAtt, testAtt), axis=0)
        if globalV.FLAGS.KEY == "APY":
            with open(globalV.FLAGS.BASEDIR + globalV.FLAGS.APYPATH + 'attribute_names.txt', 'r') as f:
                returnData.concatAttName = [line.strip() for line in f]
    elif globalV.FLAGS.SELATT == 1:
        returnData.concatAtt = np.concatenate((trainVec, valVec, testVec), axis=0)
    elif globalV.FLAGS.SELATT == 2:
        returnData.concatAtt = np.concatenate((np.concatenate((trainAtt, valAtt, testAtt), axis=0), np.concatenate((trainVec, valVec, testVec), axis=0)), axis=1)
    elif globalV.FLAGS.SELATT == 3:
        tmpAtt = np.concatenate((trainAtt, valAtt, testAtt), axis=0)
        if globalV.FLAGS.KEY == "APY":
            with open(globalV.FLAGS.BASEDIR + globalV.FLAGS.APYPATH + 'attribute_names.txt', 'r') as f:
                tmpName = [line.strip() for line in f]
            # Based on TreeMap
            tmpReduceAtt = np.zeros((tmpAtt.shape[0], 31))
            for k in range(tmpAtt.shape[0]):
                if tmpAtt[k][tmpName.index('Wing')] > 0.108:
                    tmpReduceAtt[k][0] = 1

                if tmpAtt[k][tmpName.index('Ear')] > 0.267:
                    tmpReduceAtt[k][1] = 1
                if tmpAtt[k][tmpName.index('Metal')] > 0.472:
                    tmpReduceAtt[k][2] = 1

                if tmpAtt[k][tmpName.index('Text')] > 0.016:
                    tmpReduceAtt[k][3] = 1
                if tmpAtt[k][tmpName.index('Hand')] > 0.272:
                    tmpReduceAtt[k][4] = 1
                if tmpAtt[k][tmpName.index('Tail')] > 0.709:
                    tmpReduceAtt[k][5] = 1

                if tmpAtt[k][tmpName.index('Stem/Trunk')] > 0.253:
                    tmpReduceAtt[k][6] = 1
                if tmpAtt[k][tmpName.index('Wood')] > 0.17:
                    tmpReduceAtt[k][7] = 1
                if tmpAtt[k][tmpName.index('Ear')] > 0.645:
                    tmpReduceAtt[k][8] = 1
                if tmpAtt[k][tmpName.index('Glass')] > 0.002:
                    tmpReduceAtt[k][9] = 1

                if tmpAtt[k][tmpName.index('Screen')] > 0.421:
                    tmpReduceAtt[k][10] = 1
                if tmpAtt[k][tmpName.index('Cloth')] > 0.176:
                    tmpReduceAtt[k][11] = 1
                if tmpAtt[k][tmpName.index('2D Boxy')] > 0.11:
                    tmpReduceAtt[k][12] = 1
                if tmpAtt[k][tmpName.index('Tail')] > 0.354:
                    tmpReduceAtt[k][13] = 1
                if tmpAtt[k][tmpName.index('Foot/Shoe')] > 0.424:
                    tmpReduceAtt[k][14] = 1

                if tmpAtt[k][tmpName.index('Furn. Seat')] > 0.283:
                    tmpReduceAtt[k][15] = 1
                if tmpAtt[k][tmpName.index('Handlebars')] > 0.78:
                    tmpReduceAtt[k][16] = 1
                if tmpAtt[k][tmpName.index('Exhaust')] > 0.01:
                    tmpReduceAtt[k][17] = 1
                if tmpAtt[k][tmpName.index('Saddle')] > 0.04:
                    tmpReduceAtt[k][18] = 1

                if tmpAtt[k][tmpName.index('Clear')] > 0.025:
                    tmpReduceAtt[k][19] = 1
                if tmpAtt[k][tmpName.index('Furn. Seat')] > 0.678:
                    tmpReduceAtt[k][20] = 1
                if tmpAtt[k][tmpName.index('Headlight')] > 0.283:
                    tmpReduceAtt[k][21] = 1
                if tmpAtt[k][tmpName.index('Snout')] > 0.864:
                    tmpReduceAtt[k][22] = 1
                if tmpAtt[k][tmpName.index('Ear')] > 0.884:
                    tmpReduceAtt[k][23] = 1

                if tmpAtt[k][tmpName.index('Taillight')] > 0.101:
                    tmpReduceAtt[k][24] = 1
                if tmpAtt[k][tmpName.index('Round')] > 0.163:
                    tmpReduceAtt[k][25] = 1
                if tmpAtt[k][tmpName.index('Tail')] > 0.463:
                    tmpReduceAtt[k][26] = 1
                if tmpAtt[k][tmpName.index('Occluded')] > 0.334:
                    tmpReduceAtt[k][27] = 1

                if tmpAtt[k][tmpName.index('Exhaust')] > 0.069:
                    tmpReduceAtt[k][28] = 1
                if tmpAtt[k][tmpName.index('Metal')] > 0.074:
                    tmpReduceAtt[k][29] = 1
                if tmpAtt[k][tmpName.index('Saddle')] > 0.243:
                    tmpReduceAtt[k][30] = 1
            returnData.concatAtt = tmpReduceAtt


    # Check where there is some class that has same attributes
    returnData.allClassName = trainClass + valClass + testClass
    if globalV.FLAGS.HEADER == 1:
        print('\nCheck matching classes attributes')
        for i in range(returnData.concatAtt.shape[0]):
            for j in range(i + 1, returnData.concatAtt.shape[0]):
                if np.array_equal(returnData.concatAtt[i], returnData.concatAtt[j]):
                    print('{0} {1}: {2} {3}'.format(i, returnData.allClassName[i], j, returnData.allClassName[j]))
        print('')


    # Shuffle and split Training data
    trX70 = trY70 = trAtt70 = None
    trX30 = trY30 = trAtt30 = None
    np.random.seed(globalV.FLAGS.SEED)
    sf = np.arange(trainX.shape[0])
    np.random.shuffle(sf)
    trainX = trainX[sf]
    for z in range(0, len(trainClass)):
        eachInputX = []
        eachInputY = []
        eachInputAtt = []
        for k in range(0, trainX.shape[0]):
            if trainY[k] == z:
                eachInputX.append(trainX[k])
                eachInputY.append(trainY[k])
                eachInputAtt.append(returnData.concatAtt[trainY[k]])
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
    returnData.trX70 = trX70
    returnData.trY70 = trY70
    returnData.trAtt70 = trAtt70
    returnData.trX30 = trX30
    returnData.trY30 = trY30
    returnData.trAtt30 = trAtt30


    # Balance training class
    baX70 = baY70 = baAtt70 = None
    sampleEach = 1000
    for z in range(len(trainClass)):
        eachInputX = []
        eachInputY = []
        eachInputAtt = []
        for k in range(0, trX70.shape[0]):
            if trY70[k] == z:
                eachInputX.append(trX70[k])
                eachInputY.append(trY70[k])
                eachInputAtt.append(trAtt70[k])
        eachInputX = np.array(eachInputX)
        eachInputY = np.array(eachInputY)
        eachInputAtt = np.array(eachInputAtt)
        if eachInputX.shape[0] > sampleEach:
            if baX70 is None:
                baX70 = eachInputX[:sampleEach]
                baY70 = eachInputY[:sampleEach]
                baAtt70 = eachInputAtt[:sampleEach]
            else:
                baX70 = np.concatenate((baX70, eachInputX[:sampleEach]), axis=0)
                baY70 = np.concatenate((baY70, eachInputY[:sampleEach]), axis=0)
                baAtt70 = np.concatenate((baAtt70, eachInputAtt[:sampleEach]), axis=0)
        else:
            duX70 = np.copy(eachInputX)
            duY70 = np.copy(eachInputY)
            duAtt70 = np.copy(eachInputAtt)
            while duX70.shape[0] < sampleEach:
                duX70 = np.concatenate((duX70, eachInputX), axis=0)
                duY70 = np.concatenate((duY70, eachInputY), axis=0)
                duAtt70 = np.concatenate((duAtt70, eachInputAtt), axis=0)
            if baX70 is None:
                baX70 = duX70[:sampleEach]
                baY70 = duY70[:sampleEach]
                baAtt70 = duAtt70[:sampleEach]
            else:
                baX70 = np.concatenate((baX70, duX70[:sampleEach]), axis=0)
                baY70 = np.concatenate((baY70, duY70[:sampleEach]), axis=0)
                baAtt70 = np.concatenate((baAtt70, duAtt70[:sampleEach]), axis=0)
    returnData.baX70 = baX70
    returnData.baY70 = baY70
    returnData.baAtt70 = baAtt70


    # Validation classes
    sf = np.arange(valX.shape[0])
    tmp = list()
    for i_0 in range(valY.shape[0]):
        tmp.append(returnData.concatAtt[valY[i_0] + len(trainClass)])
    returnData.vX = valX[sf]
    returnData.vY = valY[sf] + len(trainClass)
    returnData.vAtt = np.array(tmp)[sf]


    # Test classes
    sf = np.arange(testX.shape[0])
    tmp = list()
    for i_0 in range(testY.shape[0]):
        tmp.append(returnData.concatAtt[testY[i_0] + len(trainClass) + len(valClass)])
    returnData.teX = testX[sf]
    returnData.teY = testY[sf] + len(trainClass) + len(valClass)
    returnData.teAtt = np.array(tmp)[sf]


    return returnData

if __name__ == "__main__":
    argumentParser()

    createFolder()

    dataSet = loadDataset()

    if globalV.FLAGS.HEADER == 1:
        print('Shuffle Data shape')
        print(dataSet.trX70.shape, dataSet.trY70.shape, dataSet.trAtt70.shape)
        print(dataSet.trX30.shape, dataSet.trY30.shape, dataSet.trAtt30.shape)
        print(dataSet.vX.shape, dataSet.vY.shape, dataSet.vAtt.shape)
        print(dataSet.teX.shape, dataSet.teY.shape, dataSet.teAtt.shape)
        print(dataSet.baX70.shape, dataSet.baY70.shape, dataSet.baAtt70.shape)


    if globalV.FLAGS.OPT == 1:
        print('\nTrain Attribute')
        attModel = attribute()
        # attModel.trainAtt(dataSet.baX70, dataSet.baAtt70, dataSet.vX, dataSet.vAtt, dataSet.teX, dataSet.teAtt)
        attModel.trainClass(dataSet.baX70, dataSet.baY70, dataSet.trX30, dataSet.trY30)

    elif globalV.FLAGS.OPT == 2:
        model = attribute()
        outFile = open("tmp.csv", "a")

        def calNN (xx, yy, comb=False):
            tmpAtt_1 = model.getEmbedded(xx)
            nbrs = NearestNeighbors(n_neighbors=11, algorithm='ball_tree').fit(tmpAtt_1)
            distances, indices = nbrs.kneighbors(tmpAtt_1)
            print(tmpAtt_1.shape)
            print(indices.shape)

            if comb:
                sumV = 0
                for i in range(3133):
                    pos = [yy[j] for j in indices[i]]
                    sumV += ((np.sum(pos == pos[0]) - 1) / 10.0) * 100
                print(sumV / float(3133))
                outFile.write('{0},'.format(sumV / float(3133)))

                sumV = 0
                for i in range(3133, 4122):
                    pos = [yy[j] for j in indices[i]]
                    sumV += ((np.sum(pos == pos[0]) - 1) / 10.0) * 100
                print(sumV / float(4122-3133))
                outFile.write('{0},'.format(sumV / float(4122-3133)))

                sumV = 0
                for i in range(4122, indices.shape[0]):
                    pos = [yy[j] for j in indices[i]]
                    sumV += ((np.sum(pos == pos[0]) - 1) / 10.0) * 100
                print(sumV / float(indices.shape[0]-4122))
                outFile.write('{0},'.format(sumV / float(indices.shape[0]-4122)))

                sumV = 0
                for i in range(indices.shape[0]):
                    pos = [yy[j] for j in indices[i]]
                    sumV += ((np.sum(pos == pos[0]) - 1) / 10.0) * 100
                print(sumV / indices.shape[0])
                outFile.write('{0},'.format(sumV / indices.shape[0]))

            else:
                sumV = 0
                for i in range(indices.shape[0]):
                    pos = [yy[j] for j in indices[i]]
                    sumV += ((np.sum(pos==pos[0])-1) / 10.0)*100
                print(sumV/indices.shape[0])
                outFile.write('{0},'.format(sumV/indices.shape[0]))
            print('')


        calNN(dataSet.trX30, dataSet.trY30)
        calNN(dataSet.vX, dataSet.vY)
        calNN(dataSet.teX, dataSet.teY)

        tmpCombX = np.concatenate((dataSet.trX30, dataSet.vX, dataSet.teX), axis=0)
        tmpCombY = np.concatenate((dataSet.trY30, dataSet.vY, dataSet.teY), axis=0)
        calNN(tmpCombX, tmpCombY, comb=True)
        outFile.write('\n')
        outFile.close()