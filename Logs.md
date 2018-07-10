# 100DaysOfML

Pledged to the #100daysofMLCode. This will serve as the Journal for the entire 100 Days.

## Objectives

* Learn what ML is and its implications to society.
* Create 1 project per week related to ML and upload it on Github.
* Participate in the community and learn from others in the same journey.

## Note to self

Have fun and more importantly learn to give 1-2 hours a day for this cause. I already work 8+ Hours a day for others. 1-2 hours dedicated for this is relatively small.:joy: Learn by doing, have fun, collaborate.

>"Get as much education as you can. Nobody can take that away from you" -Eben Upton

## Start of the logs

### Day 1: July 7, 2018

First of all what is Machine Learning?
From reddit's [ELI5:Machine Learning](https://www.reddit.com/r/explainlikeimfive/comments/4v3u4l/eli5_what_is_machine_learning/) what I learned is that machine learning is a way to program a computer in a way that it would be able to figure something out without you typing it in **rule based, multiple if statements**. You would then **TRAIN** the machine to provide an output by feeding it data. The more data you have the more the machine can get the context of what you are trying to achieve. As it feeds on more data the more complex the things it can come up with.

Reading an article from [Medium](https://medium.com/@lampix/machine-learning-ml-and-its-impact-on-humanity-71d041298ac) about ML's impact on humanity. It is very exciting to know what machine learning is changing on the fields we know today. Imagine how different and exciting it would be when we reach the point where the things our machine learning models discover today becomes the input to the machine learnings we are about to have. The possibilities. Its exciting. :grinning:

### Day 2: July 8, 2018

Continuing now on the Intro to Machine Learning course by Udacity. I started this when I was in Thailand last June 2. I am already in Lesson 2: Naive Bayes but I had to pause because my Python was still crappy, it still is but I have finished the Intro to Python and I promise to code and practice more every day. With this initiative, I sure am going to be coding for a while. :smiley:

Anyway, the lesson for today is Gaussian Naive Bayes. The documentation as well as sample code for [Gaussian Naive Bayes in SKlearn](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) is in this link.  What is Naive Bayes anyway? From [Medium](https://medium.com/@gp_pulipaka/applying-gaussian-na%C3%AFve-bayes-classifier-in-python-part-one-9f82aa8d9ec4) it says:
>Primarily Naive Bayes is a linear classifier, which is a supervised machine learning method and works as a probabilistic classifier as well.

The article was very technical for me to wrap my head around. So I went to reddit [ELI5](https://www.reddit.com/r/explainlikeimfive/comments/1p4r3e/eli5_naive_bayes_classifier/). So Naive Bayes is a **Classifier**, this is the first thing. From the reddit answer:
>classifiers is a way to use **MATH** to identify something.

Now on to the Bayes part. For readings on [Bayes](https://en.wikipedia.org/wiki/Thomas_Bayes). Basically, Bayes was the guy who formulated the theories of statistics and probability we now know as [Bayes' Theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem). *Side note: This was also the hardest section of the Mathematics exam for the Boards.  :laughing: :laughing: :laughing:*
Going deeper into Bayes' theorem you will get the concept that:
>Bayes' Theorem can give probability of an event, based on prior knowledge of conditions related to the event

The closest example I can think of here that could be related is duck-typing. If it loos like a duck, walks like a duck and quacks like a duck, then it probably is a duck. That goes without saying that you NEED to have prior idea of how a duck walks, looks and quacks.

I have also tried the example outlined in the SKlearn documentation. The code is in 'GNB-SKLearn.py'.

### Day 3: July 9, 2018

I am currently having multiple projects at once. The Deep Learning nanodgree in [Udacity](https://www.udacity.com/course/deep-learning-nanodegree--nd101) is starting this July 10 and I am enrolled. This #100DaysOfMLCode would be a great supplement to track my progress in the nano-degree. I enrolled because I am becoming more interested to pursue the field of AI. I am interested in knowing more and I would like to enter the field and contribute. This would be part of my life long learning initiative. This was paid with the money I saved up as *Education Funds*:thumbsup:

In terms of progress for today, I was able to continue watching UD120. It is helpful but I also found the [crash course](https://developers.google.com/machine-learning/crash-course/) in Google Developers helpful. Although I have just finished watching the Intro and Framing, if the course structure is the same then I would recommend this course better than the one in Udacity. The crash course has some reading parts which I can follow along which I **-PERSONALLY-** prefer.

Reading over the crash course, there is a prerequisites page and a poll on your current background in ML. Since I am starting from zero background I have to go over the entire course in the order it is placed. Additionally, there are some recommended prerequisites to the course to better aid the understanding and the pace of the course:

* Intro-level Algebra - Which was already covered in college so I just need to review  :tada:
* Proficiency in programming basics, and some experience coding in Python -  :muscle: should be manageable since I have some experience in coding and I just finished the Intro to Python Course :satisfied:. I might need to grasp the Tensorflow workflow but that should be manageable.

I also modified the SKlearn example on Gaussian Naive Bayes. This might seem like I am all over the place by starting up a lot but once I get into the flow and focus on the course this journal would have more structure. To circle back to the modification in SKlearn code, I tried adding a sort of story on the data, the idea is that the data collected were taken from the test for drug use. More on this is already documented in the 'GNB-SKlearn.py' code.

**What I learned today?**
Basic terminology for (supervised) machine learning. The first term is **Labels**. Simply put, it is the thing that we are predicting. For a Spam email filter its going to be "Spam" or "Not Spam". For gender classifier, "Male" or "Female". For demographics it could be "Teens", "Children", "Adult", "Elderly". Its the **TRUE** value for the given *features* we have.

The next term to discuss is **Features**. Its the input variable for the system. It is used to define a label we have. For example, for a dog breed classifier it could be "Spotted" or "Unspotted", "Big" or "Small" build, "Nose length" can be a feature. The set of feature can vary from single featured data to millions of features for complicated data.

An **Example** is a particular instance of a data, **x**. Since **x** can contain one or many sets inside it, we can consider it as a **vector** that is why its in boldface. A basic example can be classified into:

* **Labeled** examples which include the feature/s and the corresponding labels
> labeled examples: {features, label}: (x, y)
Labeled data is used to **train** the model we are trying to create. In the ducktyping example we can use the features that we know of the duck like its sound, and color and we can label it as either a duck or not a duck based on the **actual** value.

* **Unlabeled** data on the other hand contains features but not the label.
> unlabeled examples: {features, ?}: (x, ?)
Once we have trained the model from labeled data, we can now input an unlabeled data to the model and know the label of the data.

**Models** are what defines the relationship between features and labels. Think of it as the black box between the input and output:smile:. It is here that the relationships and connections are built and reinforced with data. There are *two* phases for a model to go through:

* First is obviously **Training** the model or the model **Creation**. This is where the model *crunches* the data and starts building the relationship web inside it. The vector of features and label are inputs initially in this phase so that the model can build relationships.
* After the training comes the **Inference** phase. Obviously we want to use our trained model to predict or infer from new unlabeled data a possible label. This is where the value of training comes forward. The more data you have the better your model becomes and the better your model becomes obviously would yield a more accurate **inference**.

What is the difference between **Regression** and **Classification**?
A **classification** model predicts discrete values. The example was that a classification model would predict if an email is "Spam" or "Not spam", it can predict the breed of a dog "German Shepherd" "Bulldog" "Boxer" etc.
A **Regression** model predicts continuous values. For example "probability of being pregnant", "Future value of a stock" etc.

Moving forward on the crash course you get to **Linear Regression**.

A familiar topic considering this was already discussed in ENGSTAT during uni. From what I remember, the data would be plotted in a scatter plot and then there is a *best fit line* that would be available from the labeled data provided. Now recalling from algebra the formula for a line is *y = mx + b* where m is the slope of the line or gradient, b is the y-intercepts (y @ x = 0). In ML the names change as well as the information they correspond to. y becomes y' which is the label prediction. m becomes w to correspond to the *weight* of the feature (x). b is is still be but it now denotes *bias*. So the linear regression equation becomes **y' = wx + b**. Now this is only true for *simple linear regression* models where we have one feature and one label. We can have multiple features which in turn has multiple weights so a more sophisticated linear regression equation could look like this: **y' = w1x1 + w2X2 + ... + wnxn + b**.

In the crash course the example given was the relationship between the chirps of a cricket and the temperature of the environment the cricket is in. The relationship can be plotted as a simple linear regression with the number of chirps in direct proportion (positive weight) with the temperature.

Once the model has been created for the linear regression via the labeled data we can already **infer** a data based on the model.

**Training** a model for ML means finding the best weights for each feature with the goal of minimizing the over all error for the given training data. **Weight** is the coefficient of a feature in a linear model, it can also be called **edge** for deep network. The goal of training is find the ideal weight for a given feature. Training needs to find the ideal weight because it wants to minimize the error between the model which is the *best fit line* and the actual training data. In supervised learning, this process of finding weights to minimize loss is called **empirical risk minimization**.

Now we need to define what a loss is? Simply put the **loss** is the penalty for the model when a bad prediction is made. There are two types discussed, one is **L1** loss which is the absolute value difference between the actual value and the predicted value. The other is **L2** loss which is the squared loss, it is the square of the absolute value difference between actual value and predicted value: *(y_actual - y')<sup>2</sup>*. In linear regression models L2 loss is used and the reason is that L2 reacts strongly to outliers in the data.

Mean Square Loss or (MSE) takes into consideration the entire data set. Its the average squared loss per example in the data set. So basically its the __summation of *(y_actual - y')<sup>2</sup>* / N__ where __N__ is the total number of data points in the set.

This is the end of Day 3 update. I am still going to watch some of Siraj's video on ML and see if I can hack/code some of his examples tonight. Will post about the updates tomorrow.

TODO:

* Udacity's DL nano-degree introductions
* Reducing loss topic in Crash course
* Essence of Linear Algebra (3Blue1Brown) Chapters 3 and 4 in 1.5X speed. :muscle:

### Day 4: July 10, 2018

Expecting a light load today. Will review Linear Algebra by 3Blue1Brown. I am targeting chapters 3 and 4 to start. After this, I will continue on the Crash course for the Reducing Loss topic (advertised as a 60 min. exercise). There is a webinar today at 8PM I think? If I remember it correctly its 6:00 PM Pacific time.

So *Essence of Linear Algebra* from 3Blue1Brown. So i-hat and j-hat from vectors do have a name **basis vectors**. I am denoting them for now as i and j. So i and j can be considered as the unit vector for the axis, i for the x-axis and j for the y-axis and k for z-axis. So the review materials during the ECE boards now make more sense :smiley:. If we consider i and j as unit vectors then a coordinate with _(3,-5)_ could become *3i - 5j*. So the x and y-coordinates now become **scalar** and they scale the unit vectors by their value. In case of x the unit vector is scaled 3 times and in terms of y the unit vector is scaled 5 times and then flipped (due to the negative). Now that makes sense. By using basis vectors we were able to transform our coordinate into a sum of scaled vectors.

Scaling two vectors and adding them is called _linear combination_, so _3i - 5j_ is a linear combination (did I get that right?). The _span_ of i and j are the set of all their possible linear combinations.

Vectors vs. Points
The gist is that _when dealing with a single vector then an arrow representation is fine and in dealing with sets of vectors then the point representation is enough_.

The concept of the span of a vector becomes interesting when there are three-dimensions. For _one dimension_ we were able to get a *line*. With _two dimensions_ we **can** get two results, one is a *line* when the second unit vector lies on top of the first, the other is a *plane* when the two unit vectors lie on different axis. In _three dimensions_ we can get three possible results. We can still get the line and the plane but the additional result for the span we can get is that of a _cube_. Do note that to get a span we just scale the unit vectors to all possible value (positive and negative), So there is now the question on why is there going to be a unit vector lying on top of another? Yes, it is _redundant_ in a sense that it does not add anything to our span. The term for this is __linearly dependent__, and the definition is that a vector is linearly dependent if it can be expressed as the linear combination of the other and this would be true iff it is in the span of the other vector. When we say that a vector is __linearly independent__ then that vector has to add to the span. Think of it as adding a new dimension to your span. If its one dimension then there can be no dependence since there is no redundancy. For the two dimensional vector the span with a single line would mean that the vector is linearly dependent. If it can add a new dimension, say by allowing the span of the first vector (a line) to move in another axis then it adds to the span which in this case now becomes a plane (since every point in the line can be moved in the axis provided by the second vector).
> If we do not add a new _span_ for each other, then we are __linearly dependent__. We are just _redundancies_ of one another. If this lesson gives you a new perspective of your previous knowledge then it is __linearly independent__, and its add a new dimension to what you can do, your _span_.

There is a quiz at the end of Chapter 2.

> Technical definition: The __basis__ of a vector space is a set of _linearly independent_ vectors that _span_ the full space.

The question was , why does this definition makes sense?
It basically means that the basis of a vector space is the set of unit vectors that add to the span. For example we have a 3-D vector of i,j and k. The most span we can get is when the basis is (i, j, k) which gives us a cube for a span. If, say, i and j happen to coincide then the span is reduced to a plane since we are actually able to move at 2 dimensions. The basis this time becomes ((i, k) or (j, k)) this happened because i and j are just redundancies so each can still become a basis considering that only one of them is in the set. The pairing (i, k) and (i, j) would still give you linearly independent vectors.

Next lesson is **Linear Transformation** and __Matrices__.

First to be discussed is the word _transformation_ which in this case means a function. We enter a value and we get its output. The output is _"transformed"_ due to a given _function_. The idea of using the word transform instead of function stems from the fact that we can imagine vectors _"moving" or "transforming"_ due to the function.
The word _linear_ is also clarified. For a transformation to be considered linear it has to meet two properties:

* All lines must remain lines without getting curved.
* The origin must remain in place.

So if you think about it, linear transformation can be considered as the manipulation of the vector in a way that the values move but that the form stays the same. One test you can do is to input coordinates in the original plane and try to see if the line formed is still a line once you transform it. If you want to visualize linear transformation, then the output grid lines would still be parallel and evenly spaced.

The succeeding discussion takes more graphical processing at first since the animation of the transformation can be difficult to follow at first. But there was a technique that was taught which is to always consider only where the value of the basis vectors moved in relation to the original grid. Just remembered that this was covered in VECANAL in college. WOW. But getting back to the topic, why do we need to understand the technique? Because it makes it easier to understand the transformation. I find it hard to put the rest in writing but here is the try:

Consider this, we have a vector 3i + j. Substituting this into basis vectors we can get the pairing (3, 1). consider we apply a transform to that point where it now lands in a shifted grid. We can just get the values of the place where i and j landed relative to the original grid. By this means we can get a 2 x 2 matrix where the first column is for the value of i (in terms of x and y) and j (also in x and y). Its hard to do it in markdown form but consider the first pair for where the i landed as (a, b) and where j landed as (c, d) we can then get the 2 x 2 matrix. To get the transform of the original 3i + j we just have to multiply it with the new 2 x 2 matrix representation of the basis vectors in the transformed grid. I can hardly understand what I wrote so its recommended to just watch the video. :joy:

The second video was about Linear transformations and matrices. First there was a recap about what it means to be linear: *lines remain lines* and *the origin does not move*. Then there is a review on the transformation where we take the new location of i and j as the representation of that transform. Now on to matrices, simply put matrices are just coordinate representation of a vectors location. Consider the following: _v = 2i + 3j_. If we apply a transform to v that it becomes _v'_ then we can get an equation _v' = 2i' + 3j'_. Do note that this is made possible because _2 and 3_ in this case is scalar. Now back to the original idea behind a transform using unit vectors, suppose we want to _shear_ the vector _v = 2i + 3j_. We first get the values of the _i' and j' unit vectors_ when shear is applied and represent them as a matrix with values [ 1 1 > 0 1], note that > means new line. # Note to self: I have to practice formatting in .md.:sad:

Why is representing the shear or any other transformation in terms of basis vector movements important? Its because we can then apply matrix multiplication to get the new value of a vector after the transformation. Back to the example, _v = 2i + 3j_ we can just represent this as [ 2 > 3 ]. We now have a scalar pair of [ 2 > 3 ] and the unit vector of [ 1 1 > 0 1]. Recall that scaling is simply stretching or _multiplying_ the scale to the unit vector, so we can  _apply the transformation_ via multiplication of the _scalar_ and the _unit vector_ matrices. That would mean [ 2 > 3 ] [ 1 1 > 0 1] which leads us to the transform of the original vector as [ 4 > 3] which we can then write back in terms of unit vectors as _v' = 4i' + 3j'_.

After hammering down on the concept of transformation we then proceed with the properties of transformation. we take note that order matters in matrices. AB != BA. Visualize this, A is shear and B is 90deg rotation. Applying shear first then rotating would give you a different value that applying rotation first and shear. We do a transform (multiplication) right to left even if we read it from left to right. Say A = shear and B = rotation. AB is rotation and shear when transformed, quite tricky.

Now with the concept of how to apply matrix transforms in check, we now prove that matrix transforms are associative. A(BC) == (AB)C, remembering in college when I used to compute this, proving this was time consuming. But now, knowing that multiplication is simply the transformation of matrices, you can already tell that it should not matter how its computed as long as the order of transformation is the same its going to produce the same result. _Now obviously, the professor would not accept an essay form to a math question because LHS == RHS is always the way. :sigh:_

### Day 5: July 11, 2018

Setting up my Anaconda. Updated my MS VS code to 64-bit. Its the start of the nanodegree today. Immediately encountered an issue with Jupyter being blocked by the network settings.

### Current Resources

[Udacity's FREE course Intro to Machine Learning](https://classroom.udacity.com/courses/ud120)

[Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/)