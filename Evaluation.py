from dill import dill
from CommonMethods import processImage, generateAllContourPointsForClustering, imageFileDirPath, dataSaveDirPath
from ConsoleOutMethods import showClusters
from KmeansClassification import kmeansClassification


def diagramEvaluation():
    images = []
    images.append(processImage(imageFileDirPath, 'proj_diag1_Mixed.jpg'))
    with open(dataSaveDirPath + 'trained_data.pkl', 'rb') as f:
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
