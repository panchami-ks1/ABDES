################################################################
####### Automate Block Diagram Evaluation System (ABDES) #######
############## Project Execution begins from here ##############
################################################################

# Imports
from ArrowDetection import arrowEvaluation
from Evaluation import diagramEvaluation
from Training import trainData

if __name__ == "__main__":
    images_names_to_train = ["proj_diag1_Mixed.jpg"]
    image_name_for_evaluation = "proj_diag1_Mixed.jpg"
    trainData(images_names_to_train)
    diagramEvaluation(image_name_for_evaluation)
    arrowEvaluation()
