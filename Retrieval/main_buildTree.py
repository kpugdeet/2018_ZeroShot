import graphviz
import argparse
import os
import numpy as np
from sklearn import tree

"""
import created files
"""
from loadData import loadData

"""
Global Variable
"""
import globalV

def argumentParser():
    """
    Get all argument values from command line.
    :return: No return, set value to globalV variable.
    """
    parser = argparse.ArgumentParser()

    # Input image size & attribute dimension
    parser.add_argument('--width', type=int, default=227, help='Image width')
    parser.add_argument('--height', type=int, default=227, help='Image height')

    # Dataset Path
    parser.add_argument('--BASEDIR', type=str, default='/media/dataHD3/kpugdeet/', help='Base folder for dataset and logs')
    parser.add_argument('--AWA2PATH', type=str, default='AWA2/Animals_with_Attributes2/', help='AWA2 dataset')
    parser.add_argument('--CUBPATH', type=str, default='CUB/CUB_200_2011/', help='CUB dataset')
    parser.add_argument('--SUNPATH', type=str, default='SUN/SUNAttributeDB/', help='SUN dataset')
    parser.add_argument('--APYPATH', type=str, default='APY/attribute_data/', help='APY dataset')
    parser.add_argument('--GOOGLE', type=str, default='PRE/GoogleNews-vectors-negative300.bin', help='Google Word2Vec model')
    parser.add_argument('--KEY', type=str, default='APY', help='Choose dataset (AWA2, CUB, SUN, APY)')
    parser.add_argument('--DIR', type=str, default='APY_Tree', help='Choose working directory')
    parser.add_argument('--numClass', type=int, default=32, help='Number of class')
    globalV.FLAGS, _ = parser.parse_known_args()

def createFolder():
    """
    Check Folder exist.
    :return: None
    """
    if not os.path.exists(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR):
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR)
        os.makedirs(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup')

def buildTree():
    """
    Load data set and split to Train, Validation, and Test set.
    Choose whether to use attribute, word2Vec, or attribute+word2Vec to train attribute and classify model.

    :return:
    """
    (trainClass, trainAtt, trainVec, trainX, trainY, trainYAtt), \
    (valClass, valAtt, valVec, valX, valY, valYAtt), \
    (testClass, testAtt, testVec, testX, testY, testYAtt) = loadData.getData()

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

    concatAtt = np.concatenate((trainAtt, valAtt, testAtt), axis=0)
    allClassName = trainClass + valClass + testClass
    if globalV.FLAGS.KEY == "APY":
        with open(globalV.FLAGS.BASEDIR + globalV.FLAGS.APYPATH + 'attribute_names.txt', 'r') as f:
            concatAttName = [line.strip() for line in f]

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(concatAtt, range(len(allClassName)))
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=concatAttName,
                                    class_names=allClassName,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render(globalV.FLAGS.KEY+'_Tree')


if __name__ == "__main__":
    argumentParser()
    createFolder()
    buildTree()