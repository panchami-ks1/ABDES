################################################################
####### Automate Block Diagram Evaluation System (ABDES) #######
############## Project Execution begins from here ##############
################################################################

# Imports
from Evaluation import diagramEvaluationNew
from SVMImplementation import predictFromTrainData
from Training import trainData

if __name__ == "__main__":
    images_names_to_train = ["diag1.jpg"]
    image_name_for_evaluation = "diag1.jpg"             # Success full matching case.
    # image_name_for_evaluation = "diag3_excess.jpg"        # All the points matched success with some extra unwanted points/arrows.
    # image_name_for_evaluation = "diag4_less.jpg"          # Some points or arrow is missing prom the input image
    trainData(images_names_to_train)
    diagramEvaluationNew(image_name_for_evaluation)
    #predictFromTrainData()

