import argparse
import numpy as np
import random
import sys
import os
from PIL import Image
import math

def preprocess_image(image):
    # Convert to grayscale
    gray_image = image.convert('L')

    # Resize the image
    resized_image = gray_image.resize((28, 28))

    # Convert the image to a numpy array
    numpy_image = np.array(resized_image)

    # Normalize the image
    normalized_image = numpy_image / 255.0

    return normalized_image

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probabilities = {}
        self.feature_probabilities = {}
        self.classes = []
        self.features = []

    def train(self, train_images, train_labels, k):
        self.classes = list(set(train_labels))
        self.features = range(len(train_images[0]))
        # Calculate class probabilities
        total_samples = len(train_labels)
        for class_label in self.classes:
            class_samples = [img for img, label in zip(train_images, train_labels) if label == class_label]
            class_probability = len(class_samples) / total_samples
            self.class_probabilities[class_label] = class_probability
            # Calculate feature probabilities for each class
            class_samples = np.array(class_samples)
            feature_probabilities = {}
            for feature in self.features:
                feature_samples = class_samples[:, feature]
                feature_probability = self.calculate_feature_probability(feature_samples, k)
                feature_probabilities[feature] = feature_probability
            self.feature_probabilities[class_label] = feature_probabilities

    def calculate_feature_probability(self, feature_samples, k):
        unique_values, counts = np.unique(feature_samples, return_counts=True)
        total_samples = len(feature_samples)
        num_unique_values = len(unique_values)
        probabilities = {}

        for value in range(16):  # Assuming feature values range from 0 to 15
            value_count = counts[np.where(unique_values == value)]
            if len(value_count) == 0:
                value_count = 0
            else:
                value_count = value_count[0]
            probability = (value_count + k) / (total_samples + (num_unique_values * k))
            probabilities[value] = probability

        return probabilities

    def predict(self, test_images):
        predictions = []
        for image in test_images:
            max_log_probability = float('-inf')
            predicted_class = None
            for class_label in self.classes:
                class_log_probability = math.log(self.class_probabilities[class_label])
                feature_probabilities = self.feature_probabilities[class_label]

                image_log_probability = 0
                for feature, value in zip(self.features, image):
                    feature_probability = feature_probabilities[feature].get(value, 1e-10)
                    image_log_probability += math.log(feature_probability)

                class_log_probability += image_log_probability

                if class_log_probability > max_log_probability:
                    max_log_probability = class_log_probability
                    predicted_class = class_label

            predictions.append(predicted_class)
        return predictions


    def calculate_normal_probability(self, x, mean, std):
        if std == 0:
            if x == mean:
                return 1
            else:
                return 0
        else:
            exponent = np.exp(-(np.power(x - mean, 2) / (2 * np.power(std, 2))))
            return (1 / (np.sqrt(2 * np.pi) * std)) * exponent


def setup_arg_parser():
    parser = argparse.ArgumentParser(description='Learn and classify image data.')
    parser.add_argument('train_path', type=str, help='path to the training data directory')
    parser.add_argument('test_path', type=str, help='path to the testing data directory')
    parser.add_argument("-o", metavar='filepath',
                        default='classification.dsv',
                        help="path (including the filename) of the output .dsv file with the results")
    return parser

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

def loadData(path, trueSym):
    tIm, tLab = [], []
    for dir, _, files in os.walk(path):
        for fName in files:
            if(fName.endswith(".png")):
                img_path = os.path.join(dir, fName)
                label = trueSym[fName]
                tIm.append((np.array(Image.open(img_path)).astype(int)//170).flatten())
                tLab.append(label)
    return tIm, tLab

def loadTestData(path):
    tImgs, tFnames = [], []
    for dir, _, files in os.walk(path):
        for fName in files:
            if(fName.endswith(".png")):
                img_path = os.path.join(dir, fName)
                tImgs.append((np.array(Image.open(img_path)).astype(int)//170).flatten())
                tFnames.append(fName)
    return tImgs, tFnames

def string_is_num(string):
    num_str=['0','1','2','3','4','5','6','7','8','9']
    if(string in num_str):
        return True
    else:
        return False

def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    print('Training data directory:', args.train_path)
    print('Testing data directory:', args.test_path)
    print('Output file:', args.o)
    print("Running Naive Bayes classifier")

    true_symbols = getTrueSymbols(getPath(args.train_path))

    # load train data and verification data
    train_images, train_labels = loadData(args.train_path, true_symbols)

    how_many = [0] * 10
    for label in train_labels:
        if(string_is_num(label)):
            how_many[int(label)] += 1
    max_num_img = max(how_many)
    for label in range(len(how_many)):
        duplicate_count = 0
        while how_many[label] < max_num_img:
            for i in range(len(train_labels)):
                if int(train_labels[i]) == label:
                    train_images.append(train_images[i])
                    train_labels.append(train_labels[i])
                    how_many[label] += 1
                    duplicate_count += 1
                    if duplicate_count >= max_num_img:
                        break


    # load test data
    test_images, test_f_names = loadTestData(args.test_path)
    #for various data sets various k can be helpfull
    k_array = [0.5]
    for k in range(len(k_array)):
        # Initialize and train Naive Bayes classifier
        classifier = NaiveBayesClassifier()
        classifier.train(train_images, train_labels, k_array[k])

        # Predict the classes for test images
        predictions = classifier.predict(test_images)

        #Write the results to the output file
        with open(args.o, "w") as file:
            for f_name, predicted_class in zip(test_f_names, predictions):
                file.write(f"{f_name}:{predicted_class}\n")


if __name__ == "__main__":
    main()