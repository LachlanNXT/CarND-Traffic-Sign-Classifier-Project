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
| RELU					|												|
| Softmax				| used for training        									|

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


