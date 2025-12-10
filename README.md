# Abstract

Sign Vision is a set of machine learning models that will extract posture information and hand coordinates from videos of American Sign Language (ASL) to translate them into complete English sentences. SignVision will use a combination of Google MediaPipe Hand Landmarking model and CMU Open Pose model. The overall purpose of this project is to translate ASL into English for people who don’t understand sign language.

# Project Overview

- By: Zhening Hu & Vamsi Garghi
- Project Purpose: The overall purpose of this project is to translate ASL into English for people who don’t understand sign language.
- Motivation: 
  - To break the communication barrier.
  - explore the ASL language to its full extent
  - Better interactivity between deaf and hard of hearing people to communicate more      independently.
 - Methods
   - How2Sign dataset (34gb) is taken and a training model is run to get features for the training model.
   - Features are extracted from the dataset such as left & right hand cordinates,         Torso & Limb cordinates and tokenized labels.
   - These features are used to created the training dataset (.pt file) and           created the vocab meta file for english translations.
   - The training dataset is then used to create sequence modeling
   - Using the sequence modeling we will output english text based on video and its information. 

# Dataset Information

## Raw Data file Structure
<img width="657" height="266" alt="image" src="https://github.com/user-attachments/assets/d279f7d9-69e8-4f5c-8075-2cadd24fdabe" />

- Our Model is trained on a pose-based representation derived from the How2Sign dataset, which provides parallel ASL videos and English sentence-level annotations. Each sample in our dataset consists of:
 - An ASL video clip of a signer.
  <img width="398" height="343" alt="image" src="https://github.com/user-attachments/assets/40887eac-94a3-4804-aa4a-e8f490275466" />
 
 - A corresponding English sentence from the How2Sign csv files (used as the target text   for translation).
 - <img width="1090" height="45" alt="image" src="https://github.com/user-attachments/assets/1f671199-9d9c-4fc1-855b-8de07b2eadf4" />


## Data Preprocessing Stage

- Convert the green screen videos into different features such as:
 - Left and right hand landmarks such as x, y coordinates and associated confidence scores. (Mediapipe)
 - Upper-body keypoints (e.g., shoulders, elbows, wrists) and their confidence scores. (MMPose)

# Architecture

## Model Architecture Encoder - Decoder 

### Encoder

- Encoder: 1D Convolutional Layer across frames to learn relationship between frames.
- Stack of 6 transformer encoder layers to learn a representation of the coordinate information in each frame and how this information changes across frames (from 1D CNN).

 ### Decoder 
 
- Decoder: Stack of Transformer Decoder Layers with cross attention to generate next token using coordinates representation learned from encoder.
- Input to decoder: take tokenized sentence label and shift sequence to the right, for example:
- Actual token sequence:
    (w1, ..., wN, [SEP])
- Shifted token sequence:
    ([CLS], w1, ..., w_{N-1}, [SEP])
- Each decoder layer also use cross attention to use coordinate information in predicting new token.
- Query from decoder, key and value from encoder.
- Output of the decoder is mapped to tokenizer dimensions using a final linear layer.
- Objective Function: Cross Entropy Loss between decoder output and label tok.

### Cross-Attention

- Allows the decoder to “ask” the encoder which parts of the pose information is relevant to predicting words.
- The encoder will respond with pose information, this is how our model relates visual data to text.

<img width="551" height="308" alt="image" src="https://github.com/user-attachments/assets/9169b7da-4e0d-48dc-aeed-31e94b2750ea" />

### Model Architecture Diagram

<img width="595" height="627" alt="image" src="https://github.com/user-attachments/assets/cae457c5-281b-43f2-bcf8-91b9b10cd7f0" />


# Results Evaluation

## Model output with predicted sentence:

![Predicted sentence Label](output/model_predicted_sentence.png)

## Actual reference sentence:

![Raw sentence Label](output/Raw_sentence.png)

## ROGUE - 1

- Measures the number of words in the raw sentence that is present in the predictive sentence.
- Number of Overlapping words are: 21/27 
Recall-Oriented Understudy for Gisting Evaluation (ROUGE 1): 0.778

<img width="600" height="419" alt="image" src="https://github.com/user-attachments/assets/aaadf832-3073-4a46-b4c6-7a86b2597cb0" />


## BLEU - 1

- Measures the number of words in the predictive sentence that is present in the raw sentence.
- ARound 21 words present so the Bilingual Evaluation Understudy (BLEU - 1): 0.772

<img width="613" height="437" alt="image" src="https://github.com/user-attachments/assets/e0d40d5a-21a0-4b61-8d1b-5232fb4d1f8b" />


# Repo Structure
```bash
METCS777-Term-Project-Team6/
│
├── .idea/
│   ├── inspectionProfiles
│   ├── .gitignore
│   └── METCS777-Term-Project.iml
│   └── misc.xml
│   └── modules.xml
│   └── vcs.xml
│
├── code/
│   └── __pycache__
│   └── init.py
│   └── compare.py
│   └── data.py
│   └── inference.py
│   └── lab.ipynb
│   └── landmarker_demo.py
│   └── model.py
│   └── pretrained_model.py
│   └── tokenizer.py
│   └── train.py
│   └── transforms.py
│   └── utils.py
├── data/
│   ├── english_csv
│       └── how2sign_realigned_train.csv 
│   ├── precomputed_train
│       └── sample_00000.pt
│       └── ....
│       └── sample_01002.pt
│       └── vocab_meta.pt
│   ├── pretrained_model
│       └── checkpoint
│          └── rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth
│       └── mmpose_config
│          └── __init__.py
│          └── default_runtime.py
│          └── rtmpose_m_8xb256-420e_coco-256x192.py
│       └── hand_landmarker.task
├── output/
│   ├── {output}.png
├── .gitignore
├── README.md
├── requirements.txt

```


# Installation & Environment Setup

## Clone the Repo
``` git clone SSH
cd folder
```

## 

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
- train.py as the primary script. 
- The raw video input directory as the second argument.
- The output data folder for the training data to land.


## How to run the code

### Locally
The full raw video dataset is not included in the GitHub repository.
However, precomputed training data is available under:
data/pre_trained_data/


### Training

1. Activate the correct Anaconda environment
2. Ensure dependencies are correctly installed.
3. Run the training script:
   
``` bash 
python train.py --feature_dir data/pre_train_data --model_type gru --batch_size 8 --epochs 5
```

### Testing

1. Activate the correct Anaconda environment
2. Locate the path to trained model file and testing video file
3. Run the testing script in terminal: python inference.py --video_path path-to-test-file --checkpoint pretrained-model-file --max_frames 300 --frame_subsample 2 --num_keypoints 17 --max_decode_len 60

