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

### Training
1. Activate the correct Anaconda environment:
2. Ensure dependencies are correctly installed.
3. Run the training script: python train.py --feature_dir data/pre_train_data --model_type gru --batch_size 8 --epochs 5

### Testing
1. Activate the correct Anaconda environment
2. Locate the path to trained model file and testing video file
3. Run the testing script in terminal: python inference.py --video_path path-to-test-file --checkpoint pretrained-model-file --max_frames 300 --frame_subsample 2 --num_keypoints 17 --max_decode_len 60


# Model Performance Results

## Raw Data file

<img width="657" height="266" alt="image" src="https://github.com/user-attachments/assets/d279f7d9-69e8-4f5c-8075-2cadd24fdabe" />


# Dataset and Results Overview

Model output with predicted sentence:

![alt text](https://github.com/victorhuzhening/METCS777-Term-Project-Garghi-Hu/data/for_display/output1.png)

![alt text](https://github.com/victorhuzhening/METCS777-Term-Project-Garghi-Hu/data/for_display/output2.png)

Actual reference sentence:

![alt text](![alt text](https://github.com/victorhuzhening/METCS777-Term-Project-Garghi-Hu/data/for_display/output_label.png))


## Dataset Overview

Our Model is trained on a pose-based representation derived from the How2Sign dataset, which provides parallel ASL videos and English sentence-level annotations. Each sample in our dataset consists of:
  An ASL video clip of a signer.
  A corresponding English sentence from the How2Sign TSV files (used as the target text   for translation).

## How We Use The Videos

### MediaPipe Hands 

Detects Left and right hand landmarks (x, y coordinates and associated confidence scores).

### MMPose

detects Upper-body keypoints (e.g., shoulders, elbows, wrists) and their confidence scores.

## Results Explanation

### Model Setup

Two main Architecture: GRU based Seq-to-Seq Model & Transformer Encoder-decoder

### Example Review

Input video states: “The next area we’re going to talk about is rhythm.”
Model output states: “i' m going to be talking about rhythm”

This showcases some key properties for our model such as:
  1. The Model correctly identifies the topic of Rhythm
  2. The output is not a word for word translation of the reference but a paraphrase         that preserves the idea
  3. The generated sentence is fluent but somewhat informal and contains minor               formatting issues (e.g., "i' m" instead of "I’m").
