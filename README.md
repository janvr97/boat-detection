## Content of Repository

Given the nature of my varied work method, I made a summary of what my workflow would look like in a single script, called **main.py**. Note, however, that this is not the script that I executed to obtain all results at once: this is only a condensed overview of the resulting optimal workflow I decided upon after learning from an iterative search process.

I assembled most of the many different scripts I used and altered along the process in the folder **Old Scripts**, so that a notion of my iterative workflow can be obtained.

I uploaded this all at once on Github since I'm quite new to the platform, so I'm still learning how to properly make use of it, instead opting to first focus on actually building the right model.

## Workflow

The building of a working model trained to detect boats in a video in real time was an iterative process of trying out multiple different models and methods in order to together accomplish the following sequence of steps:

1. **Download the dataset** of images from Kaggle
2. **Create annotations** for all these images
3. **Split** the images and annotations in a training and validation set
4. Build or **acquire a model** fitted for the specific task of detecting multiple types of boats and drawing bounding boxes in real-time on a video, trained on a dataset of 1500 boats of 9 different classes
5. **Train** the model on this dataset and validate its performance
6. Download a **test video**
7. Perform **real-time detection** on this video

I first tried to let the annotations be created automatically, but eventually decided it would be best to create them myself using labelImg. 

Then I wrote a script to split the images and corresponding annotations in a training and test set. 

I first tried detection using ResNet, but after not getting the expected results I opted for the pre-trained YOLOv5 model, taking the nature of the task, the size of the dataset and the amount of classes into consideration. 

After downloading this model, I performed training using the built-in training script for 50 epochs on the training and validation set.

To confirm the trained model's capability, I downloaded several representative videos and let the model detect the boats in real-time.

## Implementation

The downloading of the images, creating of annotations and putting them into their respective folders, I did **manually**.

To split the images and annotations into a training and test set and put them in their respective "images" and "labels" folder, I wrote a **script**.

To download test videos, I did so either manually and used a **script**.

For downloading the pre-trained model, for training this model and for letting it detect (in real-time) on a video, I used direct commands in the **command terminal**.

## Resources

I supplemented my **own knowledge** with the use of **ChatGPT** for giving me ideas and advice, as well as correct my code and provide (better) suggestions. Besides this, I also frequently scoured the **internet** for help and ideas.






