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

So now, I am able to get it to work. Yay :smile:. I am currently trying out a simple classifier with tensorflow. Encountered an issue with the tensorflow not installed by default in Anaconda. Made a mistake of following the installation guide in Tensorflow about adding tensorflow to anaconda. I added a new environment instead of adding tensorflow to the root environment :sigh:. I was not able to find a doc detailing how to install somthing on the root env. I am guessing I have to do a pip install via gitbash or cmd prompt to the root folder and nobody got time for that. Actually was able to get the tensorflow module(?) added to the root environment via the anaconda gui.

Anyway, I was able to move and try more examples for the pre-trained deep learning models in the nano-degree. Quite fun actually. The Deep-traffic simulator which was a neural net for self-driving cars was particularly fun. There is also a Flappy bird model which I have not yet tried. There are also recommended readings for the course. I am interested in getting the flappy bird model to work. I will definitely circle back to it.

Moving on to the next lesson, we have an introduction to Anacondas. In the introduction video we get an explanation on why we use Anaconda. One thing that struck me was that Anaconda is used since it has pre-loaded libraries that are useful. The other reason was that it allows us to make use of the Conda environment manager. Now why do we need an environment manager? The explanation is that so we can control the version of our modules and our interpreter. One possible upside here is that we can be sure that all our modules actually work together. The other thing is that we can add the portability of our code through sharing by actually having all the dependencies for our project in one virtual environment. Well now it makes sense to me, before in Pycharm I actually just use one single virtual environment, the one with the most modules and if there are missing modules I just add them. Knowing the importance of environment control, I now appreciate its value and its utility.

Why anaconda? Anaconda has functionalities that make it easier to get started in data science. It comes pre-loaded with packages that are usually used for data science like numpy, pandas and sklearn. With Anaconda there is also a package and environment manager that comes with it called __conda__. Again as I have found out, the utility of being able to manage your environment per project allows us less stress in dealing with compatibility issues or allows us to use an older version of a library that might have a functionality that is no longer available on newer versions.

## Package Managers: conda and pip

Package managers like __pip__ allows us to add software and libraries to our computer. __Conda__ is the same as pip in a sense that it also allows us to install and manage our softwares, the main difference is that conda is generally geared towards data science packages pre-installed while pip is a more general purpose. conda allows us to install and manage software across multiple platforms as _conda is not python specific_. Another point to take note of is that _there will be times when we need to use pip to install a package._

Some useful conda commands to install packages

> conda install <package_name>

Package managers like pip and conda are smart enough to also install the required dependencies needed by the package you want to install. For example, scipy would need numpy to run. Entering _conda install scipy_ would not just install scipy but also get numpy for you as well.

To add multiple packages:

> conda install <package_name1> <package_name2> ... <package_nameN>

To remove a package:

> conda remove <package_name>

To update a package:

> conda update <package_name>

To update all packages in the current environment:

> conda update --all

To see the list of packages in the environment:

> conda list

To search for a package:

> conda search \*package name\*
> conda search '\*package name\*' # To escape the wildcard
> google search :joy:

## Environment managers: conda

__Environment__ is pretty straight forward. It allows isolation of packages for individual projects. They allow us to contain the scope of a package only to that project. Why would this be useful? There would be some projects where the functionality we want is available only in older versions of a package, then you will work on a project that also requires the same package but in the most recent version. Obviously the older version needed for the former project gets updated and that will lead you to issues later on when you try to go back and forth between the two projects. If we use environments in this setting we can tailor fit our package requirements in a way that best suits or needs. _Its quite similar to the cloud architecture where you only pay for what you require and you can allocate your specifications (to a certain extent) to allow you better performance_.

> _Here is an analogy I can think of for this: its similar to building a pc._  They allow us build up and spec out our PC in a very specific purpose. For example we want to create a simple PC for basic streaming, we obviously do not need a very strong GPU or a very powerful processor. Since we are in control of our specs we can opt for the most basic components that can allow us to do our intentions without going overboard in terms of computing power and budget.

To create a new environment with packages:

> conda create -n <environment_name> <[List of packages]>

To create a new environment with python:

> conda create -n <environment_name> python=<python_version>

So an example of this would be conda create -n python27 python=2.7 or conda create -n python36 python=3.6. One purpose of this is that you no longer need to install multiple instances of python in your PC manually. Actually its not true, you are still creating them but conda just does all the heavy lifting.

To activate an environment:

> activate <environment_name>

The code above is for windows. You will know the current environment that conda is using by looking at the left hand side of the terminal. The environment name would be enclosed in the parenthesis. For example if you type in _activate python36_ the new line of the terminal should show _(python36) $_.

Sample code for creating an environment with python 3.6, numpy, and pandas:

> conda create -n python36 python=3.6 numpy pandas

Note that the whitespace acts as the separator and do note that you can actually set the version of the package you want the same way you set python version number.

To see the packages inside the environment:

> conda env export

This will show you the packages of the __current__ working environment.

To export the current environment:

> conda env export > <env_name>.yaml

The line above will first list out all the files and then paste the results to the _.yaml_ file. This will allow us to send out the current working environment settings to others who wish to run the same project as ours.

To create a new environment from a yaml file:

> conda env create -f <env_name>.yaml

The code above will create a __new__ environment based on the information from the .yaml file. Take note that _the name of the environment created from loading the yaml will be the same as the environment name exported in the .yaml file_.

To list out the environments:

> conda env list

To know the current working environment, just look for the \* sign beside the environment name. When you do not activate any environment during the conda startup, the default environment would be the __base__ environment or __root__. From my workstation its called __base__.

To remove an environment:

> conda env remove -n env_name

The line above would remove the environment with name _env\_name_ from the available environments in the system. __In case its not yet clear, notice that when creating environments you do not call the env argument. But in manipulating the environments you are inside the env argument. *Just be careful of it.*__

## Best practices

### On using environments

It is good practice to setup a generic environment for the 2 versions of Python. One for Python 2 and one for Python 3. This allows you to have a general purpose environment for testing. You can then populate the environment with the basic packages for the project you want. Or you can keep it generic if you just want to practice.

### On sharing environments

When we upload or share our work and projects to others, for example in github, it is good to include a requirements.yaml file or a requirements.txt file to help users replicate the working environment you had. The requirements.txt file can be obtained via [_pip freeze_](https://pip.pypa.io/en/stable/reference/pip_freeze/) and can be handy because not everyone is using conda. 

## Further Readings for the Topic on Condas

[Conda: Myths and Misconceptions](https://jakevdp.github.io/blog/2016/08/25/conda-myths-and-misconceptions/)

[Conda documentation](https://conda.io/docs/user-guide/tasks/index.html)

### TODO

- [x] Upload to git hub [DONE]
- [x ]Setup Anaconda correctly (uhmm) :smile:
- [x]Provide a copy of the Lesson2 Jupyter notebook to the repo.

### Day 6: July 12, 2018

## Lesson 4: Jupyter Notebooks

So I just continued on the Deep Learning Nano-degree. I am now at Lesson 4. off topic for a bit, Deep Learning is a subset of machine learning. So in any case, I am still doing the #100DaysOfMLCode challenge. :happy:

Back to the Jupyter Notebooks discussion, I was able to finish it and it seems like it was a version of reused from a data science subject. It has more than enough to get you to consider switching. One of the most important point given in the lesson was that Jupyter allows us to create literate programming which was proposed by Donald Kluth. In his words:

> __Instead of imagining that our main task is to instruct a computer what to do, let us concentrate rather on explaining to human beings what we want a computer to do.__

It is a a strong argument and one that personally makes sense. It allows even those that have no idea or background about the topic the ability to follow along in the discussion of the flow of the code. Instead of giving out a big block of code, by using notebooks, you can show chunks of it and explain what is it's purpose as you go along.

> _Trivia: Jupyter came from the combination of **Ju**lia, **Pyt**hon and **R** were the first Kernels available for Jupyter back then. Jupyter was originally the iPython notebooks. It just changed because it is now supporting not just python but rather a larger collection of software._

Additional topics covered in the lesson was __*cells*__. This is the basic input space you have for a notebook. It allows you to write code or other types of text like markdown files etc.

There was also a topic on a __*Magic Keywords*__ which are python-kernel specific commands that allows you to control the notebook itself. Its useful for interactive portions of code like plots. Magic keywords always begin with either __*%*__ or __*%%*__. For % it is used for line magic and %% is used for cell magic. Magic keywords that are often used include:

> __*%matplotlib*__
> __*%time*__ or __*%% time*__
> __*%pdb*__

An interesting read: [A Neural Network in 11 Lines of code](https://iamtrask.github.io/2015/07/12/basic-python-network/) by Trask

And a possible supporting document would be [Siraj Raval's Deep Learning in 6 weeks](https://github.com/llSourcell/Learn_Deep_Learning_in_6_Weeks/) for a structured guide on learning Deep Learning.

So I just got home. I was trying to upload the markdown files with the pictures. I can't seem to figure out how they work. :sad:
I still have to figure out the latex part as well. I wonder if it would be the same if I use jupyter. But those are not the important thing right now. :triumph: :triumph: I'll just circle back here when I have some idle time. LOL.

## Matrix determinant

Recall that a transform of a vector allows you to know the location of the new vector in space. We first get the new coordinates of the unit vectors and then we scale it with the original scalars of the vector. This will lead us to the new location of the Head of the vector (if we think about it as a dot.) If we consider the area of the original vector it will also change as we transform it. The _scale of which the area changed due to a transform_ is what we consider as the determinant. The determinant is therefore a scalar. It scales the unit vector's area by its value. Here is how I picture it: _transforming the vector moves its position **scaling its length**, as we flip the location of a vector we also squish or stretch its area and **scaling of area** is the determinant_.

Moved on to the Numpy lesson in Udactiy. This is a very un-organized day.

### Day 7: July 13, 2018

Moving on to Episode 5 of the Essence of Linear Algebra. The topic was Inverse matrix, column space and null space.

The inverse of a matrix is the reverse of the transformation created by the original matrix. This is the visual approach to it. Since we say that a matrix can correspond to a transformation then its inverse is the values on which the transformation returns to its original form. For example, if the transform is to rotate the unit vectors _clockwise_, the inverse of that would be to transform the vectors _counter-clockwise_ therefore the transformations become nullified. It is the unique property in which when you apply the transformation, say A, and apply the inverse you end up where you started.

Multiplying a matrix with its inverse is equal to the matrix that does nothing. This is called the _identity matrix_. If you recall engineering algebra its the one where the diagonal of the matrix are 1's and the rest are zero. This also called the _identity transformation_.

Now on to the application of matrix which is solving for values of systems of linear equations. We should note that we can readily find a unique solution to the system of linear equations IFF the determinant of the transformation is not equal to 0. For the succeeding example I am going to call A as the transform function matrix, x as the matrix for the varialbes (i.e. [x y z]) and v as the constants. Assuming that det (A) is != 0 then we can multiply both sides by the inverse of A. Here is the original equation: Ax = v. Applying the inverse of matrix A so that only x remains on the other side we get x = v (inv A). This way we can perform the multiplication of inv A and v. This will lead us to the values corresponding to x, y, and z which are the solutions for the equation.

__Rank__  is the number of _dimensions_ in the output of a transformation. For example you the result of a transform is a line then the rank is 1, since the line is in one-dimension. If the output is a point then the rank is 0. The set of all possible outputs of your matrix is called __column space__. Basically the column space is the span of the columns of your matrix. The rank, therefore, can be considered as the number of dimensions in your column space. When the rank is as high as it can be (i.e. no squishing to a line or a point for any column) then the matrix is considered in __full rank__. If there is squishing which leads to vectors landing on the origin then we have a null space or "kernel" of the matrix. Its the space of all vectors that become null because they land on the zero vector.

Currently doing a review of the Numpy for matrices.

I have just finished the Introduction lessons in the Deep Learning Nano-degree. I admit, I still have a lot to cover like reviewing numpy and more practice. For now, I am at the introduction to Neural Networks lesson. I plan to read up first what Neural Networks are in Siraj's channel and ELI5, possibly Medium too.

I need to possibly reinstall Anaconda. I am having some issues with the environment not being detected. I believe this is caused by the location of the folder AppData being hidden by default.
I'll install Anaconda on the C drive under my name :)

### Day 8: July 14, 2018

It has been 8 days since committing to this initiative. 92 days more. A lot can happen in that span. Slowly and surely I'll get there. For now, the updates are: I was able to re-install Anaconda in My_documents this time so that there are no issues with the hidden folders path in environment. Also, I was able to eventually finish my intro part for the Deep Learning nano-degree. For now I am in the Perceptrons portion of Neural networks. Quite an interesting topic. Its the vanilla version of many deep learning models and it actually makes sense if you think of the logic behind it.

Most of the progress today would be covered in the Notebooks. Today, I was able to watch 3blue1brown Linear Algebra 5 and 6 and Neural nets. One key point given in the Neural nets intro was that the activation factor which is usually a sigmoid functions in older Neural Networks are now being replaced by ReLU, rectified linear units. The reason for the change is that for one the Sigmoid function is slower to train, the other reason pointed out was that the ReLU mimics the activation of neurons more. A brief explanation is that the neurons in the brain actually start functioning after a certain threshold which in the case of a reLU is the origin. Its quite similar to the step function 1 at y>=0 and 0 elsewhere. reLU is quite similar to this, y=x at y>=0 and 0 elsewhere. What this looks like is a linear function but rectified at the origin.

Did something really stupid today: I slipped up and watched the Saga of Tanya the evil. :smile: But its now done, its just 12 episodes in 1.5x speed. I am getting used to this playback speed. Now back to Deep learning.

### Day 9: July 15, 2018

Will continue with the course. Target today is to get to Lesson 1: Chapter 24 at least. I plan to finish Lesson 1 this week and move on to Lesson 2 since there is a project due on July 25, which is 10 days from now. I need to be fast for now so that I can make it.

Note to self: Change the activation function symbol of a sigmoid from omega to phi in the notebook. Also, I figured out why there was trouble connecting to the internet. I was on still on the proxy from work. Face palm :LOL: For now its much better, I am able to load the videos and submit answers without the connection dropping. I am continuing on with the Neural Networks lesson. I am now on the topic of Errors. Finished until chapter 24 today. Likely that I will finish this all by tomorrow and start moving on to the next lesson. Deus Vult.

### Day 10: July 16, 2018

Almost done with lesson 1. I am now at the feed forward and back propagation lectures. This can be finished today and then there will be a lab. I do have to look carefully at the entire course so that I can pace myself. I just asked my mentor for the course about my pace, I just said that I was just done with the first lesson: Neural Networks and hoping to get feedback on my pace. Hopefully we can get some response.

### Day 11: July 17, 2018

Finished part 1 of 7. I have to keep up, its supposed to be 15 days and then due. I need to focus more on the nano-degree. :sad: The good thing is that the succeeding lessons are timed at 2 hours or less. The only thing that's long is the first project: Predicting Bike sharing data which is due in 8 days. So I have to move fast. 4 more lessons to go before the project. This is exciting.

So update is that I was able to finish my first neural network. One of the most important thing I learned was how to do a vanilla version of the neural network. I still have to work on my pandas in data manipulation and numpy for data processing and mathematics. But form what I can tell, these formulas and functions will be repetitive and all that will change is the variable that you will use.

I was browsing the documentation for numpy. It was eerily similar to Matlab in college. I don't know if they were made by the same team. Anyway, went over some of the Numpy functions and WOW, it really lives up to its name. Still browsing through the built-in functions and I am currently focusing on the Linear Algebra portion, for ML and DL.

Later today I will be going over the Gradient Descent lesson. Hopefully it would not be too taxing on my brain. My perceptrons gonna get fried after this. :joy:

### Day 12: July 18, 2018

Target to finish 2nd half of implementing gradient descent lesson. This is doable, just focus and do it slow. In chunks.

And just like that Gradient Descent Lesson is done. Some important concepts tackled were how to actually apply backpropagation given multiple layers. Again, I noticed that you just have to learn the recipe and I can actually add apply it to multiple layers, although I am not sure how yet. I will figure this one out later. By this time I am planning to use the remainder of the day to make entries on my notebook. This concept has so many layers, pun intended, that its better to just draw out the concept.

Tomorrow, the objective is to cover Training Neural Networks topic. Time check, 8 days until the due of the first project, which is a prediction algorithm using bike sharing data. 15 Days for the Convolutional networks project with a Dog Breed classifier. Some great things to look forward to. Keep grinding, keep learning.

### Current Resources

[Udacity's FREE course Intro to Machine Learning](https://classroom.udacity.com/courses/ud120)

[Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/)

[Books Source](https://www.manning.com/)