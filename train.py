import os
import numpy as np
from PIL import Image
from feature import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from ensemble import AdaBoostClassifier

def get_original_features(file):
    features = []
    path = "./datasets/original/"+file
    for filename in os.listdir(path):
        img = Image.open(path + "/" + filename).convert('L')
        img = img.resize((24, 24))
        img = np.array(img)

        img_ndp = NPDFeature(img)
        this_feature = img_ndp.extract()
        features.append(this_feature)

    np.save("./datasets/"+file+"_features.npy", features)

def get_features_dataset(face_features,nonface_features):
    num_face_sample, num_face_feature = face_features.shape
    num_nonface_sample, num_nonface_feature = nonface_features.shape

    positive_label = [np.ones(1) for i in range(num_face_sample)]
    negative_label = [-np.ones(1) for i in range(num_nonface_sample)]

    positive_samples = np.concatenate((face_features, positive_label), axis=1)
    negative_samples = np.concatenate((nonface_features, negative_label), axis=1)

    features_dataset = np.concatenate((positive_samples, negative_samples), axis=0)

    np.random.shuffle(features_dataset)
    np.save("./datasets/extract_features",features_dataset)


if __name__ == "__main__":
    get_original_features("face")
    get_original_features("nonface")
    face_features = np.load("./datasets/face_features.npy")
    nonface_features = np.load("./datasets/nonface_features.npy")
    get_features_dataset(face_features,nonface_features)
    features_dataset = np.load("./datasets/extract_features.npy")
    print(features_dataset.shape,face_features.shape,nonface_features.shape)
    num_face_feature = features_dataset.shape[1] - 1

    training_size = 800
    X_train = features_dataset[:training_size, :num_face_feature]
    X_validation = features_dataset[training_size:, :num_face_feature]

    y_train = features_dataset[:training_size, -1]
    y_validation = features_dataset[training_size:, -1]
    # print(X_train.shape,y_train.shape,X_validation.shape,y_validation.shape)
    adaboost_classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),5)
    pred_y = adaboost_classifier.fit(X_train, y_train).predict(X_validation)

    with open("report.txt", "wb") as f:
        report = classification_report(y_validation, pred_y, target_names = ["nonface","face"])
        f.write(report.encode())

