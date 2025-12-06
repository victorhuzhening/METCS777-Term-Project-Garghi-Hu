# Environment setup

## Required Dependency
Head into the requirments.txt file and make sure all the dependencies match the verison present there
** this is IMPORTANT or else the project might run into issues ** 

## Cloud Environment Setup
### S3
Set up S3 the same way as Lab one and make sure the bucket is unique 

##EMR Spark
Set up Spark the same way as the Labs and make sure its pointing to the train.py file as the first argument, the raw videos as the second argument, and output folder as the third argument.

# How to run the code
##Locally
The raw data is not pushed on the github so the trained data is present in the data/pre_trained_data. To run the code locally all we have to do is be on the anaconda environemt by using this command 
### conda activate MMPoseGood ** 
once you're in the anaconda environment and have all the dependency installed, you just run this command to start the code. 
### python train.py --feature_dir data/pre_train_data --model_type gru --batch_size 8 --epochs 5 **

this will run the code and give you the results of the training and testing. 

# Results of running the code with data

# Detailed explaination of the dataset and results


