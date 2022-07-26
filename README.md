# FutureMakers2022

## Reflections
### Day 1
I relearned a lot of python, and got my hands dirty with some numpy and panda data structures. Definitely excited to learn more!

### Day 2
Notes:

Three types of data (all must be mutually exclusive) :
* Training (80%)
  * Trains the learning algorithm models
* Validation (10%)
  * Verifies which learning algorithm is the best one
  * Determines which one is the final model
* Testing (10%)
  * Tests each of the models

Two types of parameters:

* Learnable
  * The ML algorithm learns these parameters as it goes along
* Hyper-parameters
  * Parameters that we manually insert into the ML algorithm

#### Iris Data

Is a 150 x 4 2D dataset, used commonly in machine learning computing. Note that all the elements are numerical values, the default is long. 

Properties:
* Each row is a sample (test case)
* Each column is an attribute (for example, color）
* Feature
  * Given input
* Response
  * Expected output
  
### Day 3

I learned about relatively basic linear ML algorithms, such as linear regression, KMeans, and perceptrons. These algorithms are relatively limited, 
and are unable to solve problems, such as the XOR problem. That being said, however, they are still incredibly useful. For example, KMeans is pretty good for finding patterns in the data. I'm still shocked about the convenience of pre-built functions within ML libraries, making implementation of any algorithm extremely simple.

I also dipped my toes into some TensorFlow algorithms. It seems pretty basic so far, but their logistical functions have much more to offer than the simple linear ones. Below is a solution to the XOR problem.

![image](https://user-images.githubusercontent.com/60068580/178312523-befd9e49-964b-4e41-8169-9116e85376df.png)

### Day 4

We started off by learning a few TF operations. While pretty basic for low-dimension tensors, they do help a lot in convenience for higher dimensions.
**Interesting Functions:**
* convert_to_tensor( tensor) #Pretty self-explanatory
* reduce_sum(tensor, axis, keepDims, name)
  * tensor - The tensor you want to do this operation on
  * axis - The axis you are summing it along
    * 0 - Rows
    * 1 - Columns
    * And so on
  * keepDims - A boolean
  * name - The name of such an operation

### Day 5

What a wild day today！This is by far the hardest day. I dived into backpropagation along with the structure of neural networks. Now I'm beginning to understand the nuance of coding a good model for ML. It's not as simple as using a few pre-built functions. You really need to tune it to the data.

### Day 6

Today we dived in CNN's. They are much better at image analysis and recognition than your good ol' neural network. They go through several steps:

* Convolution function - extracts specific features through the use of a kernel, ends up creating a feature matrix. 
* Pooling - Reduces the number of pixels in the image. Max pooling is more popular than average pooling. It does this to reduce computational cost and also for better feature extraction
* Repeat - Repeat the above two steps
* Neural network - Now we feed the resulting layers into a neural network, where it classifies the photo. We do this through a flattening input layer

### Day 9

Today I learned some basic loss functions, such as the CCD and SCD. Not too much, but I did notice that the KDL is really bad in the model I ran in my code. I'll take a deeper look at the reason for this. I've also never heard of KDL before...

### Day 10

Today we learned more about activation functions, and now I'm beginning to understand the reason for choosing RELU. It causes NN to learn much faster, though it has a couple problems. which can be fixed through a variety of other ones, including Leaky and Elu.

### Day 12

I dived deeper into CNN networks, and really got my hands dirty with these image classification layers, especially the dropout and other regularization layers. As I was exploring the cats vs dogs dataset, I couldn't help but notice that some of these images had humans holding these pets. These must heavily impact the CNN model architecture, especially if we're talking about the real world. After all, when people take pictures of their pets, they may not be at the center of it!

### Day 13

Overfitting and underfitting are interesting concepts. It is a common theme, since in ML intermediate values are usually better than extreme ones.

