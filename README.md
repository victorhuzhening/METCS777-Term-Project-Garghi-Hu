**SignVision**
**By Vamsi Garghi, Zhening Hu**


**1.1 ABSTRACT**

SignVision is a set of machine learning models that will extract posture information and hand coordinates from videos of American Sign Language (ASL) to translate them into complete English sentences. SignVision will use a combination of Google MediaPipe Hand Landmarking model and CMU Open Pose model. The overall purpose of this project is to translate ASL into English for people who don’t understand sign language.


**1.3 REQUIREMENTS**

1.3.1 Definite Requirements
- SignVision will recognize hand signs from camera or video
- SignVision will recognize posture from camera or video
  
1.3.2 Not-decided-yet Requirements
- Whether SignVision will recognize face gestures from camera or video
- Whether SignVision will livestream the camera and output results to a website

1.3.3 Nice-to-do Requirements
- SignVision shall use hand signs and posture to construct sentences for ASL in the english language

**1.4 HOW SUCCESS WILL BE ASSESSED**

SignVision’s success will be assessed by measuring how accurately it recognizes hand signs and posture from live camera input and the result it outputs for the signs we do. The system will be considered successful if it consistently identifies signs with high accuracy (above 75%)  and reliably outputs the correct interpretation for each detected sign. 

**1.5 METHODOLOGY EXPLANATION**

According to the study “A Large-Scale Data Set and Benchmark for Understanding American Sign Language,” the dataset contains more than 25,000 annotated videos, which were evaluated using state-of-the-art sign and action recognition methods [1]. This highlights the massive scale of ASL data and demonstrates the essential role that cloud computing plays in processing and analyzing such large datasets.

The primary dataset we will use is the How2Sign Dataset, available https://how2sign.github.io/. [2]. This will be stored in Google cloud storage and the ML model will run on spark with parallel preprocessing. This directly ties into Big Data principles as without spark our dataset (which is close to 35gb) would be too large to process efficiently. 


**1.6 DATA SOURCES**

The main dataset we will be using is the How2Sign dataset which contains videos of ASL professionals signing in front of a green screen. This dataset is segmented by complete sentences and each segmented clip is annotated which will provide a benchmark for the training model [2]. 

We will obtain the data from How2Sign’s publicly available website at: How2Sign Dataset. Specifically we will use the Green Screen RGB Clips (frontal view) and English translation (original) subsets for training. The total size of the video dataset is close to 35 GB, which is infeasible for machine learning training on a singular machine.


**1.7 CONTRIBUTIONS
**
Vamsi Garghi:
Contribution 1: Finding the dataset for training
Contribution 2: Proposal Writing
Contribution 3: Work on training the AI model
Contribution 4: Work on getting dataset into Spark
Contribution 5: Paper writing, research and slides presentation.
Contribution 6: Work on coding up the project from start to finish and tracking it on Github. 

Zhening Hu:
Contribution 1; Research for topic and finding the dataset 
Contribution 2: Proposal Writing
Contribution 3: Work on training the AI model
Contribution 4: Work on coding the project and seeing what AI model would work.
Contribution 5: Paper writing, research and slides presentation.
Contribution 6: Work on coding up the project from start to finish and tracking it on Github.  


**1.8 REFERENCES FOR PROPOSAL PHASE**

- Joze, H. R. V., & Koller, O. (2019, November 20). MS-ASL: A large-scale data set and benchmark for understanding American sign language. arXiv.org. https://arxiv.org/abs/1812.01053 
- Amanda Duarte, Shruti Palaskar, Lucas Ventura, Deepti Ghadiyaram, Kenneth DeHaan, Florian Metze, Jordi Torres, and Xavier Giró-i-Nieto CVPR, 2021


Grading Criteria
**Term Project Code Sample**
Please submit your code samples, including:

1. All code files, namely METCS777-term-project-code-sample-x-Team#.xxx. Note that code should be clean and well commented.
2. All sample datasets (small sample only). 
3. A pdf, namely METCS777-term-project-code-sample-doc-Team#.pdf, including:
- Environment setup
- How to run the code
- Results of running the code with data
- Detailed explaination of the dataset and results

Grading criteria:
- (5 pts) Clean and well commented code
- (2 pts) Environment setup
- (3 pts) How to run the code
- (5 pts) Results of running the code with data
- (5 pts) Detailed explaination of the dataset and results

BONUS: +10 pts if code is committed to a Github repo with README.md as documentation. You just need to submit link to your Github repo to get the bonus without the need of submitting 1, 2, and 3 mentioned above, making sure that it is publicly viewable.

**Term Project Presentation Slides**
Please submit your slides in both .pptx and .pdf formats, namely METCS777-Term-Project-Presentation-Team#.[pptx|pdf].

Recommended content:

1. Introduction

- What the term project is all about?
- Why the topic is important?

2. Datasets:
- Details of the datasets used for the project.

3. Methodology:
+ Architecture/Design.
+ Techniques/Approaches.
  
4. Results:
- Explanation of results/findings.
 
Grading criteria:
- (5 pts) Teamwork + Communication
- (5 pts) Organization + Timing (keep it within 10 minutes)
- (10 pts) Content
- (10 pts) Demo + Q&As
