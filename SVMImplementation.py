################################################################
####### Automate Block Diagram Evaluation System (ABDES) #######
####### SVM learning phase and evaluation related metods.#######
################################################################

# Imports

from sklearn import svm
from dill import dill
from CommonDirPaths import data_save_dir_path
import itertools


def generatePermutationDatas(length):
    data = []
    input_list = [100] * length
    for x in range(length - 1):
        input_list[x] = 0
        permuted_data = set(itertools.permutations(input_list))
        permuted_data_size = len(permuted_data)
        print "Generating training data  -->  permuted ", permuted_data_size, " datas with the input list  : ", input_list
        data.append((permuted_data_size, permuted_data))
    return data


def generateXandY(length):
    X = []
    y = []
    threshold = 100 / length
    initial_score = 100

    #Adding the fully true case, [100, 100, 100, ... , 100]
    X.append([100] * length)
    y.append(initial_score)

    # Adding the fully failure case, [0, 0, 0, ..., 0]
    X.append([0] * length)
    y.append(0)

    data = generatePermutationDatas(length)

    for (data_length, data_set) in data:
        initial_score -= threshold
        for x in range(data_length):
            y.append(initial_score)

        for element in data_set:
            X.append(list(element))

    return X, y


def predictFromTrainData():
    with open(data_save_dir_path + 'trained_data.pkl', 'rb') as f:
        data = dill.load(f)

    cluster_size = len(data.clusters)
    arrow_size = len(data.arrow_cluster_map)
    length = cluster_size + arrow_size
    X = data.x
    y = data.y
    print X

    clf = svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3,
                  gamma='auto', kernel='linear', max_iter=-1,
                  shrinking=True, tol=0.001, verbose=False)
    clf.fit(X, y)

    predict_list = [100] * length

    # Testing purpose.
    # predict_list[0] = 0
    # predict_list[1] = 0
    # predict_list[2] = 0
    # predict_list[3] = 0
    # predict_list[4] = 0
    # predict_list[5] = 0
    # predict_list[6] = 0
    # predict_list[7] = 0
    # predict_list[8] = 0
    # predict_list[9] = 0
    # predict_list[10] = 0

    print predict_list
    print clf.predict([predict_list])



# Sample SVM training logic.
# Each line of X contains a training data and y as the result value for the particular training data.
# X[i] -> y[i]

def generateTrainingData():
    X = [[100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
         [100, 0, 100, 100, 100, 100, 100, 100, 100, 100],
         [100, 100, 0, 100, 100, 100, 100, 100, 100, 100],
         [100, 100, 100, 0, 100, 100, 100, 100, 100, 100],
         [100, 100, 100, 100, 0, 100, 100, 100, 100, 100],
         [100, 100, 100, 100, 100, 0, 100, 100, 100, 100],
         [100, 100, 100, 100, 100, 100, 0, 100, 100, 100],
         [100, 100, 100, 100, 100, 100, 100, 0, 100, 100],
         [100, 100, 100, 100, 100, 100, 100, 100, 0, 100],
         [100, 100, 100, 100, 100, 100, 100, 100, 100, 0]]

    y = [100, 60, 60, 60, 60, 70, 70, 70, 70, 70]

    clf = svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3,
                  gamma=0.0, kernel='linear', max_iter=-1,
                  shrinking=True, tol=0.001, verbose=False)
    clf.fit(X, y)
    print clf.predict([[100, 100, 100, 100, 100, 0, 100, 100, 100, 100]])
