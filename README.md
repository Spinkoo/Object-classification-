# Object classification 
 SVM and MLP based on  SIFT descriptors


This code requires opencv 3.4 and python >= 3.4

( so in case you don't have it , to test a video sequence example in the ResultatsVideo there is a test.py )

python test.py 

run it & it will show you an example of results by this code 


This folder has 3 codes : 

1-learning codes ( PCA and MLP or SVM )

to test it in the command line :

python learning.py datasetpath model_id

model_id : 0 for SVM and 1 for MLP

example in this folder :

python learning.py dataBinary 1

2- Test unitaire : ( takes a binary image just for test of model purposes )

python unit_test.py image_path

example in this folder : 

python unit_test.py binary.png


3- Video sequence : 

python main.py filepath

filepath is the path to the correspending video ( image sequnces )

example of this code :

python main.py b


