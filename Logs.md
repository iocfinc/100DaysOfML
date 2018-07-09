# 100DaysOfML
Pledged to the #100daysofMLCode. This will serve as the Journal for the entire 100 Days.

## Objectives:
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
>Primarily Naïve Bayes is a linear classifier, which is a supervised machine learning method and works as a probabilistic classifier as well.

The article was very technical for me to wrap my head around. So I went to reddit [ELI5](https://www.reddit.com/r/explainlikeimfive/comments/1p4r3e/eli5_naive_bayes_classifier/). So Naive Bayes is a **Classifier**, this is the first thing. From the reddit answer:
>classifiers is a way to use **MATH** to identify something.

Now on to the Bayes part. For readings on [Bayes](https://en.wikipedia.org/wiki/Thomas_Bayes). Basically, Bayes was the guy who formulated the theories of statistics and probabilty we now know as [Bayes' Theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem). *Side note: This was also the hardest section of the Mathematics exam for the Boards.  :laughing: :laughing: :laughing:*
Going deeper into Bayes' theorem you will get the concept that:
>Bayes' Theorem can give probability of an event, based on prior knowledge of conditions realted to the event

The closest example I can think of here that could be related is duck-typing. If it loos like a duck, walks like a duck and quacks like a duck, then it probably is a duck. That goes without saying that you NEED to have prior idea of how a duck walks, looks and quacks.

I have also tried the example outlined in the SKlearn documentation. The code is in 'GNB-SKLearn.py'.

### Day 3: July 9, 2018

I am currenlty having multiple projects at once. The Deep Learning nanodgree in [Udacity](https://www.udacity.com/course/deep-learning-nanodegree--nd101) is starting this July 10 and I am enrolled. This #100DaysOfMLCode would be a great supplement to track my progress in the nano-degree. I enrolled because I am becoming more interested to pursue the field of AI. I am interested in knowing more and I would like to enter the field and contribute. This would be part of my life long learning initiative. This was paid with the money I saved up as *Education Funds*:thumbsup:

In terms of progress for today, I was able to continue watching UD120. It is helpful but I also found the [crash course](https://developers.google.com/machine-learning/crash-course/) in Google Developers helpful. Although I have just finished watching the Intro and Framing, if the course structure is the same then I would recommend this course better than the one in Udacity. The crash course has some reading parts which I can follow along which I **-PERSONALLY-** prefer.

Reading over the crash course, there is a prerequisites page and a poll on your current background in ML. Since I am starting from zero background I have to go over the entire course in the order it is placed. Additionally, there are some recommended prerequisites to the course to better aid the understanding and the pace of the course:
* Intro-level Algebra - Which was already covered in college so I just need to review  :tada:
* Proficiency in programming basics, and some experience coding in Python -  :muscle: should be manageable since I have some expreience in coding and I just finished the Intro to Python Course :satisfied:. I might need to grasp the Tensorflow workflow but that should be manageable.

I also modified the SKlearn example on Gaussian Naive Bayes. This might seem like I am all over the place by starting up a lot but once I get into the rythm and focus on the course this journal would have more structure. To circle back to the modification in SKlearn code, I tried adding a sort of story on the data, the idea is that the data collected were taken from the test for drug use. More on this is already documented in the 'GNB-SKlearn.py' code.

**What I learned today?**
Basic terminolgy for (supervised) machine learning. The first term is **Labels**. Simply put, it is the thing that we are predicting. For a Spam email filter its going to be "Spam" or "Not Spam". For gender classifier, "Male" or "Female". For demographics it could be "Teens", "Children", "Adult", "Elderly". Its the **TRUE** value for the given *features* we have.

The next term to discuss is **Features**. Its the input variable for the system. It is used to define a label we have. For example, for a dog breed classifier it could be "Spotted" or "Unspotted", "Big" or "Small" build, "Nose length" can be a feature. The set of feature can vary from single featured data to millions of features for complicated data.

An **Example** is a particular instance of a data, **x**. Since **x** can contain one or many sets inside it, we can consider it as a **vector** that is why its in boldface. A basic example can be classified into:
* **Labeled** examples which include the feature/s and the corresponding labels
> labeled examples: {features, label}: (x, y)
Labeled data is used to **train** the model we are trying to create. In the ducktyping example we can use the features that we know of the duck like its sound, and color and we can label it as either a duck or not a duck based on the **actual** value.

* **Unlabeled** data on the other hand contains features but not the label.
> unlabeled examples: {features, ?}: (x, ?)
Once we have trained the model from labeled data, we can now input an unlabled data to the model and know the label of the data.

**Models** are what deines the relationship between features and labels. Think of it as the blackbox between the input and output:smile:. It is here that the relationships and connections are built and reinforced with data. There are *two* phases for a model to go through:
* First is obviously **Training** the model or the model **Creation**. This is where the model *crunches* the data and starts building the relationship web inside it. The vector of features and label are inputed initially in this phase so that the model can build relationships.
* After the training comes the **Inference** phase. Obviouly we want to use our trained model to predict or infer from new unlabled data a possible label. This is where the value of training comes forward. The more data you have the better your model becomes and the better your model becomes obviously would yeild a more accurate **inference**.

What is the difference between **Regression** and **Classification**?
A **classification** model predicts discrete values. The example was that a classification model would predict if an email is "Spam" or "Not spam", it can predict the breed of a dog "German Sheperd" "Bulldog" "Boxer" etc.
A **Regression** model predicts continuous values. For example "probability of being pregnant", "Future value of a stock" etc.

Moving forward on the crash course you get to **Linear Regression**.

A familiar topic considering this was already discussed in ENGSTAT during uni. From what I remember, the data would be plotted in a scatter plot and then there is a *best fit line* that would be available from the labeled data provided. Now recalling from algebra the formula for a line is *y = mx + b* where m is the slope of the line or gradient, b is the y-intercepts (y @ x = 0). In ML the names change as well as the informatio they correspond to. y becomes y' which is the label prediction. m becomes w to correspond to the *weight* of the feature (x). b is is still be but it now denotes *bias*. So the linear regression equation becomes **y' = wx + b**. Now this is only true for *simple linear regression* models where we have one feature and one label. We can have multiple features which in turn has multiple weights so a more sophisticated linear regression equation could look like this: **y' = w1x1 + w2X2 + ... + wnxn + b**.

In the crash course the example given was the relationship between the chirps of a cricket and the temperature of the environment the cricket is in. The relationship can be plotted as a simple linear regression with the number of chirps in direct proportion (positive weight) with the temperature.

Once the model has been created for the linear regression via the labeled data we can already **infer** a data based on the model.

**Training** a model for ML means finding the best weights for each feature with the goal of minimizing the over all error for the given training data. **Weight** is the coefficient of a feature in a linear model, it can also be called **edge** for deep network. The goal of training is find the ideal weight for a given feature. Training needs to find the ideal weight because it wants to minimize the error between the model which is the *best fit line* and the actual training data. In supervised learning, this process of finding weights to minimize loss is called **empirical risk minimization**.

Now we need to define what a loss is? Simply put the **loss** is the penalty for the model when a bad prediction is made. There are two types discussed, one is **L1** loss which is the absolute value difference between the actual value and the predicted value. The other is **L2** loss which is the squared loss, it is the square of the absolute value difference between actual value and predicted value: *(y_actual - y')<sup>2</sup>*. In linear regression models L2 loss is used and the reason is that L2 reacts strongly to outliers in the data. 

Mean Square Loss or (MSE) takes into consideraion the entire data set. Its the average squared loss per example in the data set. So basically its the __summation of *(y_actual - y')<sup>2</sup>* / N__ where __N__ is the total number of data points in the set.

This is the end of Day 3 update. I am still going to watch some of Siraj's video on ML and see if I can hack/code some of his examples tonight. Will post about the updates tomorrow.

TODO:
- Udacity's DL nano-degree introductions
- Reducing loss topic in Crash course
- Essence of Linear Algebra (3Blue1Brown) Chapters 3 and 4 in 1.5X speed. :muscle:

### Current Resources:
[Udacity's FREE course Intro to Machine Learning](https://classroom.udacity.com/courses/ud120)

[Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/)