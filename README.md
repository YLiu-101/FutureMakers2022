# FutureMakers2022

### July 2021
|           Mon          |           Tue          |           Wed          |           Thu          |           Fri          |Sat |Sun |
|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:--:|:--:|
|                        |                        |                        |           01           |           02           | 03 | 04 |
|           05           | [06](#day-1) | [07](#day-2) | [08](#day-3) | [09](#day-4) | 10 | 11 |
| [12](#Day-5) | [13](#day-6) | [14](#day-7) | [15](#day-8) | [16](#day-9) | 17 | 18 |
| [19] | [20](#day-10) | [21](#day-11) | [22](#day-12) | [23](#day-13) | 24 | 25 |
| [26](#day-14) | [27](#day-15) | [28](#day-16) | [29](#Day-17) | [30](#weeks-5--6-30072021---13082021) | [31](#weeks-5--6-30072021---13082021) |    |

### August 2021
|           Mon          |           Tue          |           Wed          |           Thu          |           Fri          |Sat |Sun |
|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:--:|:--:|
|                        |                        |                        |                        |                        |    | [01](#weeks-5--6-30072021---13082021) |
| [03](#weeks-5--6-30072021---13082021) | [03](#weeks-5--6-30072021---13082021) | [04](#weeks-5--6-30072021---13082021) | [05](#weeks-5--6-30072021---13082021) | [06](#weeks-5--6-30072021---13082021) | [07](#weeks-5--6-30072021---13082021) | [08](#weeks-5--6-30072021---13082021) |
| [09](#weeks-5--6-30072021---13082021) | [10](#weeks-5--6-30072021---13082021) | [11](#weeks-5--6-30072021---13082021) | [12](#weeks-5--6-30072021---13082021) | [13](#weeks-5--6-30072021---13082021) | 14 | 15 |
|           16           |           17           |           18           |           19           |           20           | 21 | 22 |
|           23           |           24           |           25           |           26           |           27           | 28 | 29 |
|           30           |           31           |                        |                        |                        |    |    |

## Projects

Here are some notable projects I pursued during the program:

* [California Housing Prices][https://github.com/YLiu-101/California-Housing-Reg]
* [Wildfire Prediction][https://github.com/parkerallan/FutureMakers-2022-Team-Deep-Green] (Create-a-thon)

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

In addition, we visualized the output layers of a CNN:

![image](https://user-images.githubusercontent.com/60068580/181585308-40793e73-7fe9-4d81-b272-fcea2e0a3f2b.png)


### Day 12

I dived deeper into CNN networks, and really got my hands dirty with these image classification layers, especially the dropout and other regularization layers. As I was exploring the cats vs dogs dataset, I couldn't help but notice that some of these images had humans holding these pets. These must heavily impact the CNN model architecture, especially if we're talking about the real world. After all, when people take pictures of their pets, they may not be at the center of it!

### Day 13 - 15

Overfitting and underfitting are interesting concepts. It is a common theme, since in ML intermediate values are usually better than extreme ones.

We learned a lot through these days. Here's a quick summary:

* Autoencoders compress then decompress images to create new ones
* Affective computing becomes more accurate through more dimensions
* Data filtering + processing + filling in NA values is important
* Early stopping can help a model achieve the best results
* Hyperparameter tuning is good, but if overdone, might make it "overfit" on the training


### Day 16

Steps in NLP
* Data collection
* Preprocessing - Remove noise, normalize data. 
 * Tokenization
 * Stop word removal - removing words that have no meaning, like "this", "the", etc.
 * Stemming and lemmatization
* Feature extraction - bag of words - quite inefficient, uses a large vector with mostly zeroes, TF-IDF - is better than bag of words, also takes into account the words that are not being taken into account
 * Highly recommended Word2Vec, or glove embeddings
* Model building
 * Bayesian networks
 * Maximum entropy
* Model Evaluation
 * Loss function? These are quite complex. 

NLP Concepts
* Syntax - rules governing the structure of sentences ina given language
* Semantics - what words mean

Issues with NLP

* Nonstandard words, such as sarcasm or novel words
* Meaning is context specific
* Complex sentences
* Typos in the text

Challenges

* Lexical ambiguity 
* Requires large amounts of resources

Places to find datasets (most require ~10^6 samples)
* Reddit, Twitter, other social meida platforms
* Wikipedia
* Company financial reports
* Books
* Anywhere with valid sentence sources

#### RNN Networds

Say that we have inputs $x_0, x_1, ... x_n$. We have a node that takes in x_0 and outputs h_0. The next time it takes in h_0 and x_1, combines the two, and produces h_2. The weights are multiplied with both the hiddent states and the inputs.

Issues:

* Is not able to "memorize" the earlier inputs as the number of recursions increase. Lack of long-term memory. I eat pasta sauce \[ A lot of sentences \] He eats

#### LSTM Networks

Helps solve the long-term dependency problems of RNN. Includes a memory cell. In addition to the hidden state, it has a cell state.

Has a set of gates to determine what to do with the memory at each stage.

* Output - What should I output
* Input - Opposite of forget. What new information should I take in?
* Forget - Given previous cell state, hidden, and input, what should I forget?

![image](https://user-images.githubusercontent.com/60068580/181606309-5fdff0fe-1813-4b2e-83ad-82938494ce60.png)
Source: Wikipedia

### Day 17

It has applications in physics - optics and lighting in images.

Deep Learning vs Traditional

* Higher Accuracy
* Requires a large dataset
* 

Applications:

* Image Classification - 
* Object Segmentation - What pixels are part of this object?
* Object Detection - What objects are there?
* Object Classification
 * For example self-driving cars need to track certain objects
* Pose evaluation
 * Determining which joints/positions the figure is doing


Gans:
* Automatically learns regularities --> Creates new images
* Two parts (Actor/Critic Model):
 * Generator: Learns to generate plausible data. Generated data serves as negative training examples for the discriminator. Wants to be as close to the actual data as possible
 * Discriminator: Determines difference between generator and the real example. Goal is to be better at determining which is fake and real.
* Training: 
 * Generator inputs random noise. Initially discriminator will likely do better. 
 * Back propagation to improve both. 
 
 General Pipeline:
 
 * Prepare and load the dataset
 * Preprocess data
  * Ensure data is stratified into the test/training. Make it balanced! Standardize the image to make to it of fixed size. Reshape size, zoom, other augmentation
 * Select a machine learning architecture
 * Define parameters and train the model
 * Validate the model
 * Evaluate the model with test data
 
 To create more augmented data, you can use the following object, which goes through a bunch of permutations: train_datagen = ImageDataGenerator( rescale, rotation_range,...)
 
 
 
