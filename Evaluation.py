################################################################
####### Automate Block Diagram Evaluation System (ABDES) #######
######## Diagram Evaluation related logics defined here.########
################################################################

# Imports
from dill import dill
from CommonMethods import processImage, generateAllContourPointsForClustering, image_file_dir_path, data_save_dir_path
from ConsoleOutMethods import showClusters
from KmeansClassification import kmeansClassification


def diagramEvaluation(image_name_for_evaluation):
    images = [processImage(image_file_dir_path, image_name_for_evaluation)]
    with open(data_save_dir_path + 'trained_data.pkl', 'rb') as f:
        data = dill.load(f)

    points = generateAllContourPointsForClustering(images)

    clusters = kmeansClassification(data.clusters, points)

    evaluateAnswerPoints(points, clusters)

    showClusters(clusters)


def evaluateAnswerPoints(points, clusters):
    # Checking whether the answer diagram and trained data has equal no of text blocks/regions or not.
    check = len(points) - len(clusters)

    if check == 0:
        evaluateAnswerPointsEqual(points, clusters)
    elif check < 0:
        evaluateAnswerPointsLesser(points, clusters)
    else:
        evaluateAnswerPointsGreater(points, clusters)


# To handle scenario with question diagrams having equal no of text regions/blocks.
def evaluateAnswerPointsEqual(points, clusters):
    print "Handling equal point - cluster scenario."
    count = 0
    for p in points:
        cluster = findCluster(p, clusters)
        if cluster.points[1].text == p.text:
            count += 1
    print "Score : ", count
    pass


# To handle error scenario with question diagrams having lesser text regions/blocks.
def evaluateAnswerPointsLesser(points, clusters):
    print "Handling lesser point than cluster scenario."
    count = 0
    for p in points:
        cluster = findCluster(p, clusters)
        if cluster:
            if cluster.points[1].text == p.text:
                count += 1
        else:
            count -= 1
    print "Score : ", count
    pass


# To handle error scenario with question diagrams having more text regions/blocks.
def evaluateAnswerPointsGreater(points, clusters):
    print "Handling greater point than cluster scenario."
    count = 0
    for p in points:
        cluster = findCluster(p, clusters)
        if cluster:
            if cluster.points[1].text == p.text:
                count += 1
        else:
            count -= 1

    print "Score : ", count
    pass


def findCluster(p, clusters):
    for i, cluster in enumerate(clusters):
        if p in cluster.points:
            return cluster
