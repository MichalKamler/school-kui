import argparse
import numpy as np
import os
from collections import Counter
from PIL import Image


def setup_arg_parser():
    parser = argparse.ArgumentParser(description='Learn and classify image data.')
    parser.add_argument('train_path', type=str, help='path to the training data directory')
    parser.add_argument('test_path', type=str, help='path to the testing data directory')
    parser.add_argument('-k', type=int, 
                        help='run k-NN classifier (if k is 0 the code may decide about proper K by itself')
    parser.add_argument("-o", metavar='filepath', 
                        default='classification.dsv',
                        help="path (including the filename) of the output .dsv file with the results")
    return parser

#calc dis beetween 2 img
def imgDis(img1, img2):
    return np.linalg.norm(img1-img2)

# finds the truth.dsv in train_path
def getPath(path):
    for directory, _, files in os.walk(path):
        for filename in files:
            if(filename.endswith(".dsv")):
                return os.path.join(directory, filename)
# loads the true values for each img
def getTrueSymbols(fName):
    trueSym = {}
    with open(fName, "r") as file:
        for line in file:
            line = line.strip()
            parts = line.split(":")
            key, value = parts[0], parts[1]
            trueSym[key] = value
    return trueSym

def loadTrainImages(path, trueSym):
    images, labels = [], []
    for dir, _, files in os.walk(path):
        for filename in files:
            if filename.endswith(".png"):
                img_path = os.path.join(dir, filename)            
                images.append(np.array(Image.open(img_path)).astype(int).flatten())
                labels.append(trueSym[filename])
    return images, labels

def loadTestImages(path):
    testImg, testFnames = [], []
    for dir, _, files in os.walk(path):
        for filename in files:
            if(filename.endswith(".png")):
                img_path = os.path.join(dir, filename)                
                testImg.append(np.array(Image.open(img_path)).astype(int).flatten())                
                testFnames.append(filename)
    return testImg, testFnames

def predictImg(testImgs, TrainImgs, trainLabel, k):
    predic = []
    for i, testImgOne in enumerate(testImgs):
        dis = [imgDis(trainImgOne, testImgOne) for trainImgOne in TrainImgs]
        sortIdx = np.argsort(dis)
        kNearIdx = sortIdx[:k]
        kNearLabel = [trainLabel[i]for i in kNearIdx]
        findMostLabel = Counter(kNearLabel).most_common(1)[0][0]
        predic.append(findMostLabel)
    return predic

def writePredictions(arg, testFnames, predic):
    with open(arg, 'w') as f:
        for i, filename in enumerate(testFnames):
            predictedLabel = predic[i]                
            f.write(f"{filename}:{predictedLabel}\n")

def writeWithPercentage(arg, testFnames, predic, trueSymbols):
    predicted = 0
    with open(arg, 'w') as f:
        for i, filename in enumerate(testFnames):            
            trueLabel = filename.split(".")[0][0]
            trueLabel = trueSymbols[filename]            
            predictedLabel = predic[i]            
            if(trueLabel == predictedLabel):
                predicted +=1            
            f.write(f"{filename}:{predictedLabel}\n")
    print(f"{predicted/i*100}%")

def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    print('Training data directory:', args.train_path)
    print('Testing data directory:', args.test_path)
    print('Output file:', args.o)
    
    print(f"Running k-NN classifier with k={args.k}")
    
    trueSymbols = getTrueSymbols(getPath(args.train_path))
    #k classifier
    k = args.k if args.k != 0 else 6
    
    #load images
    trainImg, trainLabel = loadTrainImages(args.train_path, trueSymbols)

    #load test img
    testImg, testFnames = loadTestImages(args.test_path)
    #predict img
    predic = predictImg(testImg, trainImg, trainLabel, k)

    # Write predictions to output file
    writePredictions(args.o, testFnames, predic)
    #writeWithPercentage(args.o, testFnames, predic, trueSymbols)

if __name__ == "__main__":
    main()
    
