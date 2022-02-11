from imageai.Prediction import ImagePrediction
import os
import cv2
import os.path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras

main_folder = os.getcwd()
execution_path=os.getcwd()

prediction = ImagePrediction()
prediction.setModelTypeAsResNet50()
prediction.setModelPath(model_path=os.path.join(execution_path, "resnet50_imagenet_tf.2.0.h5"))
prediction.loadModel()

predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "godzilla.jpeg"), result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)