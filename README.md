# Environment setup

## Dependency
Review the requirements.txt file and ensure that all installed dependencies match the specified versions. This step is critical to prevent compatibility issues or runtime errors.

## Cloud Environment Setup
### S3
Configure an S3 bucket as demonstrated in Lab 1. 
Ensure the bucket name is globally unique to avoid conflicts.
Create a separate output folder within the S3 bucket to store the generated result files.

### EMR with Apache Spark

Configure Spark as done in previous labs.
Your Spark job should reference:

train.py as the primary script
The raw video input directory as the second argument
The output data folder (S3 or local) as the third argument

# How to run the code
## Locally
The full raw video dataset is not included in the GitHub repository.
However, precomputed training data is available under:
data/pre_trained_data/

## To execute the model locally:
1. Activate the correct Anaconda environment:
2. Ensure dependencies are correctly installed.
3. Run the training script: python train.py --feature_dir data/pre_train_data --model_type gru --batch_size 8 --epochs 5

# Results of running the code with data

# Detailed explaination of the dataset and results


