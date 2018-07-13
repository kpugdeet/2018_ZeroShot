import os
import gensim
import pickle
import numpy as np
import globalV
import glob
import cv2
import scipy.io
from random import shuffle

class loadData(object):
    @staticmethod
    def resize(imageName):
        img = cv2.imread(imageName)
        h_img, w_img, _ = img.shape
        imgResize = cv2.resize(img, (globalV.FLAGS.width, globalV.FLAGS.height))
        # imgRGB = cv2.cvtColor(imgResize, cv2.COLOR_BGR2RGB)
        imgRGB = imgResize
        imgResizeNp = np.asarray(imgRGB)
        #imgResizeNp = (imgResizeNp / 255.0) * 2.0 - 1.0
        return imgResizeNp

    @staticmethod
    def resizeBB(imageName, BB):
        img = cv2.imread(imageName)
        h_img, w_img, _ = img.shape
        BB = np.array(BB).astype(np.int32)
        if BB[1] != BB[3] and BB[0] != BB[2]:
            crop_img = img[BB[1]:BB[3], BB[0]:BB[2]]
        else:
            crop_img = img
        imgResize = cv2.resize(crop_img, (globalV.FLAGS.width, globalV.FLAGS.height))
        # imgRGB = cv2.cvtColor(imgResize, cv2.COLOR_BGR2RGB)
        imgRGB = imgResize
        imgResizeNp = np.asarray(imgRGB)
        # imgResizeNp = (imgResizeNp / 255.0) * 2.0 - 1.0
        return imgResizeNp

    # Read AwA2 dataset
    @staticmethod
    def readAwA2():
        with open(globalV.FLAGS.BASEDIR + globalV.FLAGS.AWA2PATH + 'classes.txt', 'r') as f:
            allClass = [line.split()[1] for line in f]

        shuffleAllClass = list(allClass)
        shuffle(shuffleAllClass)
        trainClass = shuffleAllClass[:27]
        valClass = shuffleAllClass[27:40]
        testClass = shuffleAllClass[40:]

        # Read attribute value for each class
        trainAtt = list()
        testAtt = list()
        valAtt = list()
        with open(globalV.FLAGS.BASEDIR + globalV.FLAGS.AWA2PATH + 'predicate-matrix-binary.txt', 'r') as f:
            for _, line in enumerate(f):
                if allClass[_] in trainClass:
                    trainAtt.append(line.split())
                elif allClass[_] in valClass:
                    valAtt.append(line.split())
                else:
                    testAtt.append(line.split())
        trainAtt = np.array(trainAtt).astype(np.float64)
        testAtt = np.array(testAtt).astype(np.float64)
        valAtt = np.array(valAtt).astype(np.float64)

        # Read train image
        trainX = list()
        trainY = list()
        for _, name in enumerate(trainClass):
            for image in glob.glob(globalV.FLAGS.BASEDIR + globalV.FLAGS.AWA2PATH + "JPEGImages/" + name + "/*.jpg"):
                trainX.append(loadData.resize(image))
                trainY.append(_)
        trainX = np.array(trainX).astype(np.uint8)
        trainY = np.array(trainY).astype(np.uint8)

        # Read valid image
        valX = list()
        valY = list()
        for _, name in enumerate(valClass):
            for image in glob.glob(globalV.FLAGS.BASEDIR + globalV.FLAGS.AWA2PATH + "JPEGImages/" + name + "/*.jpg"):
                valX.append(loadData.resize(image))
                valY.append(_)
        valX = np.array(valX).astype(np.uint8)
        valY = np.array(valY).astype(np.uint8)

        # Read test image
        testX = list()
        testY = list()
        for _, name in enumerate(testClass):
            for image in glob.glob(globalV.FLAGS.BASEDIR + globalV.FLAGS.AWA2PATH + "JPEGImages/" + name + "/*.jpg"):
                testX.append(loadData.resize(image))
                testY.append(_)
        testX = np.array(testX).astype(np.uint8)
        testY = np.array(testY).astype(np.uint8)
        return (trainClass, trainAtt, trainX, trainY), (valClass, valAtt, valX, valY), (testClass, testAtt, testX, testY)

    # Read CUB dataset
    @staticmethod
    def readCUB():
        with open(globalV.FLAGS.BASEDIR + globalV.FLAGS.CUBPATH + 'classes.txt', 'r') as f:
            allClass = [line.split()[1] for line in f]

        shuffleAllClass = list(allClass)
        shuffle(shuffleAllClass)
        trainClass = shuffleAllClass[:100]
        valClass = shuffleAllClass[100:150]
        testClass = shuffleAllClass[150:]

        # Read attribute value for each class
        trainAtt = list()
        testAtt = list()
        valAtt = list()
        with open(globalV.FLAGS.BASEDIR + globalV.FLAGS.CUBPATH + 'attributes/class_attribute_labels_continuous.txt', 'r') as f:
            for _, line in enumerate(f):
                if allClass[_] in trainClass:
                    trainAtt.append(line.split())
                elif allClass[_] in valClass:
                    valAtt.append(line.split())
                else:
                    testAtt.append(line.split())
        trainAtt = np.array(trainAtt).astype(np.float64)
        testAtt = np.array(testAtt).astype(np.float64)
        valAtt = np.array(valAtt).astype(np.float64)

        # Read train image
        trainX = list()
        trainY = list()
        for _, name in enumerate(trainClass):
            for image in glob.glob(globalV.FLAGS.BASEDIR + globalV.FLAGS.CUBPATH + "images/" + name + "/*.jpg"):
                trainX.append(loadData.resize(image))
                trainY.append(_)
        trainX = np.array(trainX).astype(np.uint8)
        trainY = np.array(trainY).astype(np.uint8)

        # Read valid image
        valX = list()
        valY = list()
        for _, name in enumerate(valClass):
            for image in glob.glob(globalV.FLAGS.BASEDIR + globalV.FLAGS.CUBPATH + "images/" + name + "/*.jpg"):
                valX.append(loadData.resize(image))
                valY.append(_)
        valX = np.array(valX).astype(np.uint8)
        valY = np.array(valY).astype(np.uint8)

        # Read test image
        testX = list()
        testY = list()
        for _, name in enumerate(testClass):
            for image in glob.glob(globalV.FLAGS.BASEDIR + globalV.FLAGS.CUBPATH + "images/" + name + "/*.jpg"):
                testX.append(loadData.resize(image))
                testY.append(_)
        testX = np.array(testX).astype(np.uint8)
        testY = np.array(testY).astype(np.uint8)

        # Change class name (remove number)
        trainClass = [x.split('.')[1] for x in trainClass]
        valClass = [x.split('.')[1] for x in valClass]
        testClass = [x.split('.')[1] for x in testClass]

        return (trainClass, trainAtt, trainX, trainY), (valClass, valAtt, valX, valY), (testClass, testAtt, testX, testY)

    # Read SUN dataset
    @staticmethod
    def readSUN():
        mat = scipy.io.loadmat(globalV.FLAGS.BASEDIR + globalV.FLAGS.SUNPATH + 'images.mat')
        fileList = [x[0][0] for x in mat['images']]
        category = list()
        dataY = list()
        for tmp in fileList:
            name = tmp.split('/')
            if len(name) > 3:
                combName = name[1]+'_'+name[2]
            else:
                combName = name[1]
            dataY.append(combName)
            if combName not in category:
                category.append(combName)
        mat2 = scipy.io.loadmat(globalV.FLAGS.BASEDIR + globalV.FLAGS.SUNPATH + 'attributeLabels_continuous.mat')
        dataYAtt = np.array(mat2['labels_cv'])

        # Shuffle train/validation/test
        shuffle(category)
        trainClass = category[:580]
        valClass = category[580:645]
        testClass = category[645:]

        # Read attribute value for each class
        trainAtt = [list() for _ in range(580)]
        valAtt = [list() for _ in range(65)]
        testAtt = [list() for _ in range(72)]

        trainX = list()
        trainY = list()
        trainYAtt = list()
        valX = list()
        valY = list()
        valYAtt = list()
        testX = list()
        testY = list()
        testYAtt = list()

        for _, tmpName in enumerate(dataY):
            if tmpName in trainClass:
                pos = trainClass.index(tmpName)
                trainAtt[pos].append(dataYAtt[_])
                trainX.append(loadData.resize(globalV.FLAGS.BASEDIR + globalV.FLAGS.SUNPATH + 'images/' + fileList[_]))
                trainY.append(pos)
                trainYAtt.append(dataYAtt[_])

            elif tmpName in valClass:
                pos = valClass.index(tmpName)
                valAtt[pos].append(dataYAtt[_])
                valX.append(loadData.resize(globalV.FLAGS.BASEDIR + globalV.FLAGS.SUNPATH + 'images/' + fileList[_]))
                valY.append(pos)
                valYAtt.append(dataYAtt[_])
            else:
                pos = testClass.index(tmpName)
                testAtt[pos].append(dataYAtt[_])
                testX.append(loadData.resize(globalV.FLAGS.BASEDIR + globalV.FLAGS.SUNPATH + 'images/' + fileList[_]))
                testY.append(pos)
                testYAtt.append(dataYAtt[_])

        trainAtt = np.array(trainAtt).astype(np.float64).mean(axis=1)
        testAtt = np.array(testAtt).astype(np.float64).mean(axis=1)
        valAtt = np.array(valAtt).astype(np.float64).mean(axis=1)

        trainX = np.array(trainX).astype(np.uint8)
        trainY = np.array(trainY).astype(np.uint8)
        trainYAtt = np.array(trainYAtt).astype(np.float64)
        valX = np.array(valX).astype(np.uint8)
        valY = np.array(valY).astype(np.uint8)
        valYAtt = np.array(valYAtt).astype(np.float64)
        testX = np.array(testX).astype(np.uint8)
        testY = np.array(testY).astype(np.uint8)
        testYAtt = np.array(testYAtt).astype(np.float64)

        return (trainClass, trainAtt, trainX, trainY, trainYAtt), (valClass, valAtt, valX, valY, valYAtt), (testClass, testAtt, testX, testY, testYAtt)

    # Read APY dataset
    @staticmethod
    def readAPY():
        with open(globalV.FLAGS.BASEDIR + globalV.FLAGS.APYPATH + 'class_names.txt', 'r') as f:
            allClass = [line.strip() for line in f]

        # # Shuffle train/validation/test
        # shuffle(allClass)
        # trainClass = allClass[:15]
        # valClass = allClass[15:20]
        # testClass = allClass[20:]

        # Choose Train and Test class
        trainClass = ['person', 'statue', 'car', 'aeroplane', 'cat', 'zebra', 'sheep', 'bicycle', 'bottle', 'sofa', 'carriage', 'bird', 'pottedplant', 'tvmonitor', 'building']
        valClass = ['centaur', 'train', 'donkey', 'jetski', 'diningtable']
        testClass = ['monkey', 'bus', 'wolf', 'dog', 'horse', 'cow', 'motorbike', 'mug', 'chair', 'boat', 'bag', 'goat']

        # K-Means Classes
        # trainClass = ['person', 'statue', 'monkey', 'aeroplane', 'cat', 'zebra', 'sheep', 'bicycle', 'chair', 'sofa', 'carriage', 'bird', 'pottedplant', 'tvmonitor', 'jetski']
        # valClass = ['centaur', 'train', 'donkey', 'building', 'diningtable']
        # testClass = ['car', 'bus', 'wolf', 'dog', 'horse', 'cow', 'motorbike', 'mug', 'bottle', 'boat', 'bag', 'goat']

        # Read attribute value for each class
        trainAtt = [list() for _ in range(15)]
        valAtt = [list() for _ in range(5)]
        testAtt = [list() for _ in range(12)]

        trainX = list()
        trainY = list()
        trainYAtt = list()
        valX = list()
        valY = list()
        valYAtt = list()
        testX = list()
        testY = list()
        testYAtt = list()

        for _, inputFile in enumerate(list(['apascal_train.txt', 'apascal_test.txt', 'ayahoo_test.txt'])):
            if _ == 2:
                tmpPath = globalV.FLAGS.BASEDIR + globalV.FLAGS.APYPATH + 'ayahoo_test_images/'
            else:
                tmpPath = globalV.FLAGS.BASEDIR + globalV.FLAGS.APYPATH + 'VOCdevkit/VOC2008/JPEGImages/'
            with open(globalV.FLAGS.BASEDIR + globalV.FLAGS.APYPATH + inputFile, 'r') as f:
                for _, line in enumerate(f):
                    tmp = line.split()
                    tmpFilename = tmp[0]
                    tmpClassName = tmp[1]
                    tmpBB = tmp[2:6]
                    tmpAtt = np.array(tmp[6:]).astype(np.uint8)

                    if tmpClassName in trainClass:
                        pos = trainClass.index(tmpClassName)
                        trainAtt[pos].append(tmpAtt)
                        trainX.append(loadData.resizeBB(tmpPath + tmpFilename, tmpBB))
                        trainY.append(pos)
                        trainYAtt.append(tmpAtt)
                    elif tmpClassName in valClass:
                        pos = valClass.index(tmpClassName)
                        valAtt[pos].append(tmpAtt)
                        valX.append(loadData.resizeBB(tmpPath + tmpFilename, tmpBB))
                        valY.append(pos)
                        valYAtt.append(tmpAtt)
                    else:
                        pos = testClass.index(tmpClassName)
                        testAtt[pos].append(tmpAtt)
                        testX.append(loadData.resizeBB(tmpPath + tmpFilename, tmpBB))
                        testY.append(pos)
                        testYAtt.append(tmpAtt)

        trainAtt = [np.array(x).astype(np.float64).mean(axis=0) for x in trainAtt]
        trainAtt = np.array(trainAtt)
        testAtt = [np.array(x).astype(np.float64).mean(axis=0) for x in testAtt]
        testAtt = np.array(testAtt)
        valAtt = [np.array(x).astype(np.float64).mean(axis=0) for x in valAtt]
        valAtt = np.array(valAtt)

        trainX = np.array(trainX).astype(np.uint8)
        trainY = np.array(trainY).astype(np.uint8)
        trainYAtt = np.array(trainYAtt).astype(np.float64)
        valX = np.array(valX).astype(np.uint8)
        valY = np.array(valY).astype(np.uint8)
        valYAtt = np.array(valYAtt).astype(np.float64)
        testX = np.array(testX).astype(np.uint8)
        testY = np.array(testY).astype(np.uint8)
        testYAtt = np.array(testYAtt).astype(np.float64)

        return (trainClass, trainAtt, trainX, trainY, trainYAtt), (valClass, valAtt, valX, valY, valYAtt), (testClass, testAtt, testX, testY, testYAtt)

    # Generate vector of word
    @staticmethod
    def getWord2Vec(inputName):
        vector = list()
        model = gensim.models.KeyedVectors.load_word2vec_format(globalV.FLAGS.BASEDIR + globalV.FLAGS.GOOGLE, binary=True)
        for _, x in enumerate(inputName):
            if x == "car_interior_frontseat":
                inputName[_] = "car_interior_front_seat"
            elif x == "donjon":
                inputName[_] = "castle_keep"
            elif x == "flight_of_stairs_natural":
                inputName[_] = "flight_stairs_natural"
            elif x == "flight_of_stairs_urban":
                inputName[_] = "flight_stairs_urban"
            elif x == "forest_needleleaf":
                inputName[_] = "forest_needle_leaf"
            elif x == "lean-to":
                inputName[_] = "lean"
            elif x == "mastaba":
                inputName[_] = "eternal_house"
            elif x == "theater_indoor_procenium":
                inputName[_] = "theater_indoor_facade"
            elif x == "thriftshop":
                inputName[_] = "thrift_shop"
            elif x == "barndoor":
                inputName[_] = "barn_door"
            elif x == "videostore":
                inputName[_] = "video_store"
            elif x == "diningtable":
                inputName[_] = "dining_table"
            elif x == "pottedplant":
                inputName[_] = "potted_plant"
            elif x == "tvmonitor":
                inputName[_] = "tv_monitor"
            elif x == "aeroplane":
                inputName[_] = "Aeroplane"
        inputName = [x.replace('+', '_') for x in inputName]
        for word in inputName:
            try:
                vector.append(model[word])
            except KeyError as e:
                print(e)
                tmp = word.split('_')
                tmpModel = np.copy(model[tmp[0]])
                for x in tmp[1:]:
                    tmpModel += model[x]
                vector.append(tmpModel)
                continue
        return np.array(vector).astype(np.float64)

    @staticmethod
    def getData():
        if os.path.isfile(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/trainClass.pkl'):
            trainClass = pickle.load(open(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/trainClass.pkl', 'rb'),)
            trainAtt = pickle.load(open(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/trainAtt.pkl', 'rb'))
            trainVec = pickle.load(open(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/trainVec.pkl', 'rb'))
            trainX = np.load(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/trainX.pkl.npy')
            trainY = np.load(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/trainY.pkl.npy')

            valClass = pickle.load(open(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/valClass.pkl', 'rb'))
            valAtt = pickle.load(open(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/valAtt.pkl', 'rb'))
            valVec = pickle.load(open(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/valVec.pkl', 'rb'))
            valX = np.load(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/valX.pkl.npy')
            valY = np.load(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/valY.pkl.npy')

            testClass = pickle.load(open(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/testClass.pkl', 'rb'))
            testAtt = pickle.load(open(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/testAtt.pkl', 'rb'))
            testVec = pickle.load(open(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/testVec.pkl', 'rb'))
            testX = np.load(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/testX.pkl.npy')
            testY = np.load(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/testY.pkl.npy')

            if globalV.FLAGS.KEY == 'SUN' or globalV.FLAGS.KEY == 'APY':
                trainYAtt = np.load(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/trainYAtt.pkl.npy')
                valYAtt = np.load(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/valYAtt.pkl.npy')
                testYAtt = np.load(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/testYAtt.pkl.npy')
            else:
                trainYAtt = None; valYAtt = None; testYAtt = None
        else:
            if globalV.FLAGS.KEY == 'AWA2':
                (trainClass, trainAtt, trainX, trainY), (valClass, valAtt, valX, valY), (testClass, testAtt, testX, testY) = loadData.readAwA2()
                trainYAtt = None; valYAtt = None; testYAtt = None
            elif globalV.FLAGS.KEY == 'CUB':
                (trainClass, trainAtt, trainX, trainY), (valClass, valAtt, valX, valY), (testClass, testAtt, testX, testY) = loadData.readCUB()
                trainYAtt = None; valYAtt = None; testYAtt = None
            elif globalV.FLAGS.KEY == 'SUN':
                (trainClass, trainAtt, trainX, trainY, trainYAtt), (valClass, valAtt, valX, valY, valYAtt), (testClass, testAtt, testX, testY, testYAtt) = loadData.readSUN()
            elif globalV.FLAGS.KEY == 'APY':
                (trainClass, trainAtt, trainX, trainY, trainYAtt), (valClass, valAtt, valX, valY, valYAtt), (testClass, testAtt, testX, testY, testYAtt) = loadData.readAPY()
            trainVec = loadData.getWord2Vec(trainClass)
            testVec = loadData.getWord2Vec(testClass)
            valVec = loadData.getWord2Vec(valClass)

            pickle.dump(trainClass, open(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/trainClass.pkl', 'wb'))
            pickle.dump(trainAtt, open(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/trainAtt.pkl', 'wb'))
            pickle.dump(trainVec, open(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/trainVec.pkl', 'wb'))
            np.save(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/trainX.pkl', trainX)
            np.save(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/trainY.pkl', trainY)

            pickle.dump(valClass, open(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/valClass.pkl', 'wb'))
            pickle.dump(valAtt, open(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/valAtt.pkl', 'wb'))
            pickle.dump(valVec, open(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/valVec.pkl', 'wb'))
            np.save(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/valX.pkl', valX)
            np.save(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/valY.pkl', valY)

            pickle.dump(testClass, open(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/testClass.pkl', 'wb'))
            pickle.dump(testAtt, open(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/testAtt.pkl', 'wb'))
            pickle.dump(testVec, open(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/testVec.pkl', 'wb'))
            np.save(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/testX.pkl', testX)
            np.save(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/testY.pkl', testY)

            if globalV.FLAGS.KEY == 'SUN' or globalV.FLAGS.KEY == 'APY':
                np.save(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/trainYAtt.pkl', trainYAtt)
                np.save(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/valYAtt.pkl', valYAtt)
                np.save(globalV.FLAGS.BASEDIR + globalV.FLAGS.DIR + '/backup/testYAtt.pkl', testYAtt)
        return (trainClass, trainAtt, trainVec, trainX, trainY, trainYAtt), (valClass, valAtt, valVec, valX, valY, valYAtt), (testClass, testAtt, testVec, testX, testY, testYAtt)
