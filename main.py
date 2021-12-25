import sys

import numpy as np
import operator
import os  # 폴더로 읽어오기 위함


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])  # 배열 생성
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 집합에서의 최솟값. param 0은 column의 최솟값을 얻게 함
    maxVals = dataSet.max(0)  # 집합에서의 최댓값
    ranges = (maxVals - minVals)  # 범위
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))  # 행렬 크기 맞춰주기
    normDataSet = normDataSet / np.tile(ranges, (m, 1))  # 구성요소 나누기
    return normDataSet, ranges, minVals


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()  # 배열 소팅 후 인덱스 반환
    classCount = {}
    for i in range(k):  # 가장 짧은 거리를 투표
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=(operator.itemgetter(1)), reverse=True)
    return sortedClassCount[0][0]

def fileToMatrix(foldername):  # 폴더 이름이 들어올거임
    # path_dir = 'C:/Users/Sim/Desktop/new'
    file_list = os.listdir(foldername)

    for i in range(len(file_list)):  # 폴더이름으로 받아서 폴더 속 파일 이름들 받아오기.
        filename = foldername + '/' + file_list[i]

        f = open(filename)
        returnMat = []

        classLabelVector = []  # 파일 이름의 라벨을 가지고 올거임
        classLabelVector.append(filename[0])  # 파일 이름의 맨 앞의 애가 라벨

        index = 0
        forOneLine = []
        for line in f.readlines():
            line = line.strip()  # 개행문자 제거
            listFormLine = line.split(' ')  # 공백 단위로 자른다.
            forOneLine.append(listFormLine)  # 하나의 행으로 합친다.
            returnMat[index: ] = listFormLine[0:]  # 한 줄 끝에서 끝까지
            # classLabelVector.append(listFormLine[-1])
            index += 1

    return returnMat, classLabelVector


#제출할 떈 이거로=!! trainingDigitsFolder = sys.argv[1]
trainingDigitsFolder = "trainingDigits"

datingDataMat, datingLabels = fileToMatrix(trainingDigitsFolder)
normMat, range, minVals = autoNorm(datingDataMat)  # 정규화

####입력되는 테스트 파일 읽기#####
#제출할 떈 이거로=!! inputData = sys.argv[2]
inputData = "testDigits"
file_list = os.listdir(inputData)
returnMat = []
classLabelVector = []# 파일 이름의 라벨을 가지고 올거임 => 입력되는 test는 라벨링 필요x

for i in range(len(file_list)):  # 폴더이름으로 받아서 폴더 속 파일 이름들 받아오기.
    filename = inputData + '/' + file_list[i]

    f = open(filename)
    returnMat = []

    # classLabelVector.append(filename[0])  # 파일 이름의 맨 앞의 애가 라벨 => 입력되는 test는 라벨링 필요x

    index = 0
    forOneLine = []
    for line in f.readlines():
        line = line.strip()  # 개행문자 제거
        listFormLine = line.split(' ')  # 공백 단위로 자른다.
        forOneLine.append(listFormLine)  # 하나의 행으로 합친다.
        returnMat[index:] = listFormLine[0:]  # 한 줄 끝에서 끝까지
        # classLabelVector.append(listFormLine[-1])
        index += 1


# inputArray = np.array([float(i) for i in inputData.split()])

for i in range(len(returnMat)):
    returnMat[i] = (returnMat[i] - minVals[i]) / range[i]

for i in range(len(returnMat)):
    print(classify0(returnMat, datingDataMat, datingLabels, 3)) #여기서 3은 k개 뽑아서 다수결할 그거
