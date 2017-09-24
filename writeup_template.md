# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---

[//]: # (Image References)

[image1]: ./report_images/AfterNormColourImage.png "out1"
[image2]: ./report_images/AfterNormColourValuesExample.png "out2"
[image3]: ./report_images/Certainty0.png "out3"
[image4]: ./report_images/Certainty1.png "out4"
[image5]: ./report_images/Certainty2.png "out5"
[image6]: ./report_images/Certainty3.png "out6"
[image7]: ./report_images/Certainty4.png "out7"
[image8]: ./report_images/Certainty5.png "out8"
[image9]: ./report_images/Certainty6.png "out9"
[image10]: ./report_images/Certainty7.png "out10"
[image11]: ./report_images/Certainty8.png "out11"
[image12]: ./report_images/Extrapics.png "out12"
[image13]: ./report_images/ImageExample.png "out13"
[image14]: ./report_images/ImageLabels.png "out14"
[image15]: ./report_images/incorrect.png "out15"
[image16]: ./report_images/PreNormColourImage.png "out16"
[image17]: ./report_images/PreNormColourValuesExample.png "out17"
[image18]: ./report_images/TestExamples.png "out18"
[image19]: ./report_images/TrainingExamples.png "out19"
[image20]: ./report_images/ValidExamples.png "out20"

### Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

### Contents
This submission includes:
The Traffic_Sign_Classifier.ipynb notebook file with all questions answered and all code cells executed and displaying output.
A HTML export of the project notebook with the name report.html.
Additional images used for the project that are not from the German Traffic Sign Dataset. Found in /data/Ready/
This writeup report (markdown file)
Here is a link to my [project code](https://github.com/LachlanNXT/CarND-Traffic-Sign-Classifier-Project)

### Data Set Summary & Exploration

I used numpy to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of testing examples = 12630
* Number of validation examples = 4410
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Exploratory visualization of the dataset.

The picture below from training set, has a corresponding label of 4, which means speed limit 70:

![alt text][image13]

The label matches the image so the data seems to have been imported correctly.

![alt text][image18] ![alt text][image19] ![alt text][image20]

We can see from the histograms above that the training, validation and test datasets have a similar distribution of image types.

### Design and Test a Model Architecture

#### 1. Preprocessing the image data.

I wanted to normalise the data first, so I converted the data tupe from 'uint8' to 'int16' so the pixel values could be negative. Then I applied the pixel = (pixel - 128)/ 128 operation. The following plots show an example of the colour distribution in an image before and after normalisation:

![alt text][image17]![alt text][image2]

I decided not to grayscale because colour is an important piece of information in some traffic signs.

I decided not to augment the dataset because the model seems to work fine without and due to time constraints.

#### 2. Final model architecture

I started with the LeNet Lab model and modified it. The resulting architecture is very similar, with an extra convolutional layer, more neurons per layer, and dropout on the first convolutional layer added.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x30 	|
| RELU					|												|
| Dropout					|					50%							|
| Max pooling	      	| 2x2 stride, valid padding, outputs 15x15x30 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 13x13x60 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 6x6x60 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 4x4x120 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 2x2x120 				|
| Flattening	    | output 2x2x120 = 480  									|
| Fully connected		| input 480, output 120       									|
| RELU					|												|
| Fully connected		| input 120, output 84       									|
| RELU					|												|
| Fully connected		| input 84, output 43       									|
| RELU					|		output	LOGITS, prediction = argmax(LOGITS) 	|
| Softmax				| used for training 							|

#### 3. Training

I started with the default batch size, epochs, and other hyperparameters from the LeNet Lab. I also used the Adam Optimiser to start with. Varying the parameters, I found that the optimiser type, learning rate, and number of neurons/parameters had the biggest effect. I started by increasing the epochs until the accuracy flattened out. Then I tried changing the batch size, which seemed to have little effect. I increased the neurons per layer quite a bit, added dropout to counter overfitting due to the parameter increase, and added another convolutional layer since this dataset is more complicated than the original LeNet. These measures increased the accuracy a little. By far the biggest improvement came from changing to the Momentum Optimiser. I did this because the Adam Optimiser seemed to get to about 89% accuracy then bounce up and down from that value, indicating to me that the learning rate was too high or the optimiser was having trouble keeping the correct descent direction. After lowering the learning rate and learning that the Adam Optimiser implements learning rate decay, I tried the Momentum optimiser. I varied the learning rate and momentum parameter, and was able to achieve much faster learning and a validation accuracy of about 94%.

The final parameters were:
* Momentum Optimiser, learning rate = 0.04, momentum = 0.6
* Epochs = 15
* Batch size = 128
* Dropout = 0.5 on the first layer only during training

#### 4. Validation set accuracy

My final model results were: 
* validation set accuracy of 94% 
* test set accuracy of 93.7%

The agreement between accuracy on both validation and test sets indicates that there is very little over-fitting. The accuracy is over the requirement of 93% so it it quite accurate.

### Test a Model on New Images

#### 1. German traffic signs found on the web

Here are nine German traffic signs that I found on the web:

![alt text][image12]

Images 2, 4, 5 and 7 are taken from photographs. The rest are reference images of the signs. The photographed images will be close to what the network was trained on, and should have an accuracy close to the test accuracy. The other images are very clear and should be even better. There is nothing particularly challenging about any of these images compared to the test/validation/training sets. The labels of these images are as below:

![alt text][image14]

The images were preprocessed in the same way as the original dataset.

#### 2. Predictions on these new traffic signs

Here are the results of the prediction:

Prediction:  [ 0  4 38 17 13 25 14 40 13]
Label:       [0, 4, 38, 17, 27, 25, 14, 40, 13]
Correct?     [ True  True  True  True False  True  True  True  True]

The model was able to correctly guess 8 of the 9 traffic signs, which gives an accuracy of 89%. Given the small size of this extra dataset, this is withing statistical error of the test accuracy. However, when we examine the incorrect classification further, we find as in the image below:

![alt text][image15]

This is an interesting result. The image is of a pedestrain sign, and the prediction is for a yield sign. However, an example of a pedestrian sign from the test set looks different to this image. Further investigation on [wikipedia](https://en.wikipedia.org/wiki/Road_signs_in_Germany) reveals this sign is a version of an informative sign, and the test sign is a regulatory sign. So I have asked the network to classify a type of sign it does not know about, so a wrong answer is expected! Disregarding this results, the network scored 100%, although without any particularly challenging images.

#### 3. Model certainty

is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The following images show the top 5 softmax probabilities for classification of each of the downloaded images:

![alt text][image3] ![alt text][image4]
![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]
![alt text][image9] ![alt text][image10]
![alt text][image11]

We can see from these plots that the model is totally sure of its prediction of all of these images. Even the image it classified incorrectly shows overwhelming certainty despite being slightly less certain than the other classifications., which it has some slight uncertainty. The corresponding sign types can be seen in the figure under section 2.



