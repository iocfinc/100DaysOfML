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

## Day 1: July 7, 2018

First of all what is Machine Learning?
From reddit's [ELI5:Machine Learning](https://www.reddit.com/r/explainlikeimfive/comments/4v3u4l/eli5_what_is_machine_learning/) what I learned is that machine learning is a way to program a computer in a way that it would be able to figure something out without you typing it in **rule based, multiple if statements**. You would then **TRAIN** the machine to provide an output by feeding it data. The more data you have the more the machine can get the context of what you are trying to achieve. As it feeds on more data the more complex the things it can come up with.

Reading an article from [Medium](https://medium.com/@lampix/machine-learning-ml-and-its-impact-on-humanity-71d041298ac) about ML's impact on humanity. It is very exciting to know what machine learning is changing on the fields we know today. Imagine how different and exciting it would be when we reach the point where the things our machine learning models discover today becomes the input to the machine learnings we are about to have. The possibilities. Its exciting. :grinning:

## Day 2: July 8, 2018

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

## Day 3: July 9, 2018

I am currently having multiple projects at once. The Deep Learning nanodegree in [Udacity](https://www.udacity.com/course/deep-learning-nanodegree--nd101) is starting this July 10 and I am enrolled. This #100DaysOfMLCode would be a great supplement to track my progress in the nano-degree. I enrolled because I am becoming more interested to pursue the field of AI. I am interested in knowing more and I would like to enter the field and contribute. This would be part of my life long learning initiative. This was paid with the money I saved up as *Education Funds*:thumbsup:

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

A familiar topic considering this was already discussed in ENGSTAT during uni. From what I remember, the data would be plotted in a scatter plot and then there is a *best fit line* that would be available from the labeled data provided. Now recalling from algebra the formula for a line is $y = mx + b$ where m is the slope of the line or gradient, b is the y-intercepts (y @ x = 0). In ML the names change as well as the information they correspond to. y becomes y' which is the label prediction. m becomes w to correspond to the *weight* of the feature (x). b is is still be but it now denotes *bias*. So the linear regression equation becomes $y' = wx + b$. Now this is only true for *simple linear regression* models where we have one feature and one label. We can have multiple features which in turn has multiple weights so a more sophisticated linear regression equation could look like this: $y' = w1x1 + w2X2 + ... + wnxn + b$.

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

## Day 4: July 10, 2018

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

The second video was about Linear transformations and matrices. First there was a recap about what it means to be linear: *lines remain lines* and *the origin does not move*. Then there is a review on the transformation where we take the new location of i and j as the representation of that transform. Now on to matrices, simply put matrices are just coordinate representation of a vectors location. Consider the following: _v = 2i + 3j_. If we apply a transform to v that it becomes _v'_ then we can get an equation _v' = 2i' + 3j'_. Do note that this is made possible because _2 and 3_ in this case is scalar. Now back to the original idea behind a transform using unit vectors, suppose we want to _shear_ the vector _v = 2i + 3j_. We first get the values of the _i' and j' unit vectors_ when shear is applied and represent them as a matrix with values [ 1 1 > 0 1], note that > means new line. # Note to self: I have to practice formatting in .md. :rage:

Why is representing the shear or any other transformation in terms of basis vector movements important? Its because we can then apply matrix multiplication to get the new value of a vector after the transformation. Back to the example, _v = 2i + 3j_ we can just represent this as [ 2 > 3 ]. We now have a scalar pair of [ 2 > 3 ] and the unit vector of [ 1 1 > 0 1]. Recall that scaling is simply stretching or _multiplying_ the scale to the unit vector, so we can  _apply the transformation_ via multiplication of the _scalar_ and the _unit vector_ matrices. That would mean [ 2 > 3 ] [ 1 1 > 0 1] which leads us to the transform of the original vector as [ 4 > 3] which we can then write back in terms of unit vectors as _v' = 4i' + 3j'_.

After hammering down on the concept of transformation we then proceed with the properties of transformation. we take note that order matters in matrices. AB != BA. Visualize this, A is shear and B is 90deg rotation. Applying shear first then rotating would give you a different value that applying rotation first and shear. We do a transform (multiplication) right to left even if we read it from left to right. Say A = shear and B = rotation. AB is rotation and shear when transformed, quite tricky.

Now with the concept of how to apply matrix transforms in check, we now prove that matrix transforms are associative. A(BC) == (AB)C, remembering in college when I used to compute this, proving this was time consuming. But now, knowing that multiplication is simply the transformation of matrices, you can already tell that it should not matter how its computed as long as the order of transformation is the same its going to produce the same result. _Now obviously, the professor would not accept an essay form to a math question because LHS == RHS is always the way. :unamused:_

## Day 5: July 11, 2018

Setting up my Anaconda. Updated my MS VS code to 64-bit. Its the start of the nanodegree today. Immediately encountered an issue with Jupyter being blocked by the network settings.

So now, I am able to get it to work. Yay :smile:. I am currently trying out a simple classifier with tensorflow. Encountered an issue with the tensorflow not installed by default in Anaconda. Made a mistake of following the installation guide in Tensorflow about adding tensorflow to anaconda. I added a new environment instead of adding tensorflow to the root environment :sigh:. I was not able to find a doc detailing how to install something on the root env. I am guessing I have to do a pip install via gitbash or cmd prompt to the root folder and nobody got time for that. Actually was able to get the tensorflow module(?) added to the root environment via the anaconda gui.

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

* [x] Upload to git hub [DONE]
* [x] Setup Anaconda correctly (uhmm) :smile:
* [x] Provide a copy of the Lesson2 Jupyter notebook to the repo.

## Day 6: July 12, 2018

### Lesson 4: Jupyter Notebooks

So I just continued on the Deep Learning Nano-degree. I am now at Lesson 4. off topic for a bit, Deep Learning is a subset of machine learning. So in any case, I am still doing the #100DaysOfMLCode challenge.  :metal:

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

So I just got home. I was trying to upload the markdown files with the pictures. I can't seem to figure out how they work.  :expressionless:
I still have to figure out the latex part as well. I wonder if it would be the same if I use jupyter. But those are not the important thing right now. :triumph: :triumph: I'll just circle back here when I have some idle time. LOL.

#### Matrix determinant

Recall that a transform of a vector allows you to know the location of the new vector in space. We first get the new coordinates of the unit vectors and then we scale it with the original scalars of the vector. This will lead us to the new location of the Head of the vector (if we think about it as a dot.) If we consider the area of the original vector it will also change as we transform it. The _scale of which the area changed due to a transform_ is what we consider as the determinant. The determinant is therefore a scalar. It scales the unit vector's area by its value. Here is how I picture it: _transforming the vector moves its position **scaling its length**, as we flip the location of a vector we also squish or stretch its area and **scaling of area** is the determinant_.

Moved on to the Numpy lesson in Udactiy. This is a very un-organized day.

## Day 7: July 13, 2018

Moving on to Episode 5 of the Essence of Linear Algebra. The topic was Inverse matrix, column space and null space.

The inverse of a matrix is the reverse of the transformation created by the original matrix. This is the visual approach to it. Since we say that a matrix can correspond to a transformation then its inverse is the values on which the transformation returns to its original form. For example, if the transform is to rotate the unit vectors _clockwise_, the inverse of that would be to transform the vectors _counter-clockwise_ therefore the transformations become nullified. It is the unique property in which when you apply the transformation, say A, and apply the inverse you end up where you started.

Multiplying a matrix with its inverse is equal to the matrix that does nothing. This is called the _identity matrix_. If you recall engineering algebra its the one where the diagonal of the matrix are 1's and the rest are zero. This also called the _identity transformation_.

Now on to the application of matrix which is solving for values of systems of linear equations. We should note that we can readily find a unique solution to the system of linear equations IFF the determinant of the transformation is not equal to 0. For the succeeding example I am going to call A as the transform function matrix, x as the matrix for the variables (i.e. [x y z]) and v as the constants. Assuming that det (A) is != 0 then we can multiply both sides by the inverse of A. Here is the original equation: Ax = v. Applying the inverse of matrix A so that only x remains on the other side we get x = v (inv A). This way we can perform the multiplication of inv A and v. This will lead us to the values corresponding to x, y, and z which are the solutions for the equation.

__Rank__  is the number of _dimensions_ in the output of a transformation. For example you the result of a transform is a line then the rank is 1, since the line is in one-dimension. If the output is a point then the rank is 0. The set of all possible outputs of your matrix is called __column space__. Basically the column space is the span of the columns of your matrix. The rank, therefore, can be considered as the number of dimensions in your column space. When the rank is as high as it can be (i.e. no squishing to a line or a point for any column) then the matrix is considered in __full rank__. If there is squishing which leads to vectors landing on the origin then we have a null space or "kernel" of the matrix. Its the space of all vectors that become null because they land on the zero vector.

Currently doing a review of the Numpy for matrices.

I have just finished the Introduction lessons in the Deep Learning Nano-degree. I admit, I still have a lot to cover like reviewing numpy and more practice. For now, I am at the introduction to Neural Networks lesson. I plan to read up first what Neural Networks are in Siraj's channel and ELI5, possibly Medium too.

I need to possibly reinstall Anaconda. I am having some issues with the environment not being detected. I believe this is caused by the location of the folder AppData being hidden by default.
I'll install Anaconda on the C drive under my name :)

## Day 8: July 14, 2018

It has been 8 days since committing to this initiative. 92 days more. A lot can happen in that span. Slowly and surely I'll get there. For now, the updates are: I was able to re-install Anaconda in My_documents this time so that there are no issues with the hidden folders path in environment. Also, I was able to eventually finish my intro part for the Deep Learning nano-degree. For now I am in the Perceptrons portion of Neural networks. Quite an interesting topic. Its the vanilla version of many deep learning models and it actually makes sense if you think of the logic behind it.

Most of the progress today would be covered in the Notebooks. Today, I was able to watch 3blue1brown Linear Algebra 5 and 6 and Neural nets. One key point given in the Neural nets intro was that the activation factor which is usually a sigmoid functions in older Neural Networks are now being replaced by ReLU, rectified linear units. The reason for the change is that for one the Sigmoid function is slower to train, the other reason pointed out was that the ReLU mimics the activation of neurons more. A brief explanation is that the neurons in the brain actually start functioning after a certain threshold which in the case of a reLU is the origin. Its quite similar to the step function 1 at y>=0 and 0 elsewhere. reLU is quite similar to this, y=x at y>=0 and 0 elsewhere. What this looks like is a linear function but rectified at the origin.

Did something really stupid today: I slipped up and watched the Saga of Tanya the evil. :smile: But its now done, its just 12 episodes in 1.5x speed. I am getting used to this playback speed. Now back to Deep learning.

## Day 9: July 15, 2018

Will continue with the course. Target today is to get to Lesson 1: Chapter 24 at least. I plan to finish Lesson 1 this week and move on to Lesson 2 since there is a project due on July 25, which is 10 days from now. I need to be fast for now so that I can make it.

Note to self: Change the activation function symbol of a sigmoid from omega to phi in the notebook. Also, I figured out why there was trouble connecting to the internet. I was on still on the proxy from work. Face palm :LOL: For now its much better, I am able to load the videos and submit answers without the connection dropping. I am continuing on with the Neural Networks lesson. I am now on the topic of Errors. Finished until chapter 24 today. Likely that I will finish this all by tomorrow and start moving on to the next lesson. Deus Vult.

### Day 10: July 16, 2018

Almost done with lesson 1. I am now at the feed forward and back propagation lectures. This can be finished today and then there will be a lab. I do have to look carefully at the entire course so that I can pace myself. I just asked my mentor for the course about my pace, I just said that I was just done with the first lesson: Neural Networks and hoping to get feedback on my pace. Hopefully we can get some response.

## Day 11: July 17, 2018

Finished part 1 of 7. I have to keep up, its supposed to be 15 days and then due. I need to focus more on the nano-degree. :sad: The good thing is that the succeeding lessons are timed at 2 hours or less. The only thing that's long is the first project: Predicting Bike sharing data which is due in 8 days. So I have to move fast. 4 more lessons to go before the project. This is exciting.

So update is that I was able to finish my first neural network. One of the most important thing I learned was how to do a vanilla version of the neural network. I still have to work on my pandas in data manipulation and numpy for data processing and mathematics. But form what I can tell, these formulas and functions will be repetitive and all that will change is the variable that you will use.

I was browsing the documentation for numpy. It was eerily similar to Matlab in college. I don't know if they were made by the same team. Anyway, went over some of the Numpy functions and WOW, it really lives up to its name. Still browsing through the built-in functions and I am currently focusing on the Linear Algebra portion, for ML and DL.

Later today I will be going over the Gradient Descent lesson. Hopefully it would not be too taxing on my brain. My perceptrons gonna get fried after this. :joy:

## Day 12: July 18, 2018

Target to finish 2nd half of implementing gradient descent lesson. This is doable, just focus and do it slow. In chunks.

And just like that Gradient Descent Lesson is done. Some important concepts tackled were how to actually apply backpropagation given multiple layers. Again, I noticed that you just have to learn the recipe and I can actually add apply it to multiple layers, although I am not sure how yet. I will figure this one out later. By this time I am planning to use the remainder of the day to make entries on my notebook. This concept has so many layers, pun intended, that its better to just draw out the concept.

Tomorrow, the objective is to cover Training Neural Networks topic. Time check, 8 days until the due of the first project, which is a prediction algorithm using bike sharing data. 15 Days for the Convolutional networks project with a Dog Breed classifier. Some great things to look forward to. Keep grinding, keep learning.

## Day 13: July 19, 2018

Here is a fun [playground](https://developers.google.com/machine-learning/crash-course/reducing-loss/playground-exercise) for neural network visualization from the Google Developers [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/). Its an interactive site that allows you to visualize how Neural Networks train and come up with its decision plane.

While on commute I decided to watch the lessons in Udemy: Deep Learning A-Z course. One the positive note I was able to finish the videos from Intro to NN course with all the theory. Coming up next would be a project for a bank data. Do note that I was only able to finish the videos because of the traffic in Manila, 15 minutes for the bus to cross an intersection, then 10 minutes wait time for the train to arrive. Silver lining is I was able to finish the course, I'll take it.

Right now, I am trying to organize my notes in the notebook. Currently working on the Gradient descent topic. Will be finished today.

## Day 14: July 20, 2018

Started doing the Neural Network Training by Luis. It really pays to watch and take notes that watch and retake notes. Going over the lesson just to take notes really takes time.

About done with the back notes for Gradient descent. Target for today are: Finish the Lesson on Training Neural Networks. -- Well this did not happen. Binged on Stranger Things S2 today. Was able to finish it all. So that's out of the way. On to writing my notes. Now on multi-layered perceptrons, i.e. Hidden Layers. They are basically the same concept as the one output perceptron. The math is still the same, its just that the operations are now mostly in matrix form. One thing to note, hidden inputs are not the same as hidden layers. Think of it this way, Input Layer, Hidden Layer/s and Output layers are our layers. Inside the layers are our units, which in Udacity I think they called inputs. But the point is that, don't mix units and layers together. Also, the vector notation of $V_{ab}$ where a is the origin and b is the destination still applies in the naming convention for our matrix and matrix elements.

Also, I was able to start Section 4 of Deep Learning A-Z in Udemy. This time it was basically back to building an Artificial Neural Networks. In this course they use Theano, Tensor Flow and Keras which makes things simpler. There is a topic regarding cleaning data and how to prepare the data for entry into a Neural Network, remember that we can only use normalized or standardized data. I need to back read it in the Appendix of the Course. It starting to dawn on me that I took this on the wrong order, it should have been AI then ML then DL. But hey, you at least have to start somewhere right?

For this weekend, I am planning to build my first project in the form of the ANN from Deep Learning AZ. I am excited actually.

For now though, I need to finish my notes and the Udacity: Training Neural Networks lesson. See you tomorrow. :smile:

## Day 15: July 21, 2018

To do today:
Training Neural Networks lesson and notes. Artificial Neural Network project from Udemy. Finish Essence of Linear Algebra. Enjoy the day!

So today I was able to finish the Training Neural Networks Lesson and the note taking. I was able to complete the introduction to the Workspace at Udacity where there are GPU enabled notebooks for future tests. Right now I am setting up my dlnd environment. There is a project due in 3 days.

To give a background on the project, its a Neural network project that will come up with a model to predict the number of bikes that our bike sharing company would need given the previous data. Basically, an optimization problem. We want to limit the excess bikes that we have that are not making money and we also want to have enough bikes that we can cater to our current and projected growth of clients.

I will work on the project tomorrow, for now I am pushing my brain. I need a fresh start tomorrow. Learned a lot today. Local minima, batch and stochastic gradient descent, different activation functions (tanh and reLU), regularization. For now I am calling it a day. Tomorrow I'll resume this.

## Day 16: July 22, 2018

Doing the project today. Bike sharing model. Actually got stuck in the back propagation in the unit testing. For now I think I solved it. On to fine tuning hyperparameters. This is really tiring.

## Day 17: July 23, 2018

Still trying to train my network. Figuring out what hyperparameters to use to meet the specifications. This is really intensive and eats a lot of time. I am planning to sleep for now and proceed with training later. My plan is this: I will be methodical in searching for the hyperparameters. It looks like 30 nodes, 0.5 learnrates work. To save time, I have to watch advanced lessons in the subject so that I don't waste it waiting for the training to complete. Hopefully I will get this done by today.

And I fucked it up, big time. I was wondering why I only pass the unit test when I manually change the learning rate below to 0.5. Turns out I was supposed to actually call a variable self.lr as the learning rate. Big setback. But will continue on nonetheless. I am already in bed. I want to rest my back. :trollface: This is fun.

So final update is I was able to pass my first project in the Deep Learning Nanodegree course. Had to play around with the hyper parameters and counter-intuitively, not all learning rates should be small. My learning rate for this was 0.7 with 5 hidden nodes and 2000 iterations. I received Training loss < 0.08 and a Validation loss < 0.18 I think. I'm sure it met the specifications of the Udacity's project as I just received a response from the reviewer that the project passed.

Had multiple learnings/insights on this project. One is that not all learning rates should be small. The idea is that it should first and foremost converge to a minimum. Second is that I got the idea of why the error term for hidden layers are structured in such a way that they need to take into account the error term of the preceding layer. The answer is again due to chain rule. There is already a derivative function in the error term for the output. So multiplying that to the derivative of the activation function of the current hidden layer is in effect a chain rule in action. Also, the transpose of the matrix is sometimes key in getting the code to work (dot products or multiplications) as well as arrive at an answer that is still the same in value but of different order (transposed).

## Day 18: July 24, 2018

For now I am moving on to Sentiment Analysis. Taught by Andrew Truskk. Basically its using Neural Networks to figure out if an outlook on a topic is Positive or Negative. Basically, quantifying the qualitative.

First, a review of the Neural network concepts. Neural networks, Feed forward, Back propagation. Then on to Gradient Descent. Then reviewed Errors, Squared Errors and Mean Squared Errors. Finally, Data set splitting. Testing and Training data. Also made some notes on the insights gained on the project. Like the learning rates as well as the transpose of matrices to allow for transformation.

Now we move on to Sentiment Analysis using Neural Networks.

## Day 19: July 25, 2018

So now we are in Sentiment Analysis. This is quite fun. There is a topic on how to optimize the system and also how to curate a data set. We were taught how to frame a problem and come up with a predictive theory. Then we move on to the concept of optimization where we accept the slight decrease in accuracy for a huge increase in speed.

In other news, I was reminded of the DevOps Course by the company today. I finished the ansible and jenkins module today. Briefly, Jenkins is for automating CI/CD while ansible is used for configuration management. At least that is how I get it. I can see it benefiting the team but I HIGHLY DOUBT that the client would want it. From all the context I can get, they are pinching every penny.

Target for today is to finish sentiment analysis module. Then perform on the notebook again to cement the idea. Also, finish the Machine Learning basics template from Deep Learning A-Z. I have downloaded the intuitions section so that should be easy. Also, I am quite enjoying reading The Subtle Art of Not Giving a Fuck by Manson. Its a good read and very insightful, and fun.

So I just finished Sentiment Analysis. There are commands in the end of the lessons that were new, it was already in matplotlib. Also, encountered an error in Bokeh dependency. Just to recap, we have 6 different projects. First project is Curating the Dataset. It was basic manipulation of data to see if we can come up with our predictive theory. Basically, we want to have a blueprint of how we are going to approach the problem, what features to count as inputs and what columns can be dropped because they will not have a bearing on the output. Then we move to creating our own NN given the basic idea of our predictive theory: Which was that the appearance of words have correlation to the general sentiment of the review. After finishing Project 2: we were able to come up with a system that is 60% accurate and quite slow. Project 3 was about filtering our Signal to Noise. What we did was instead of considering the number of times a word appears, we just take into consideration if they appear or not and we tag them as either 1 or 0. Project 4 was about further optimizing the system by tweaking the insides of our NN. Instead of doing a dot product of ALL the matrices, we simply looked for the index of the words in the current review and we update the weights for those indices. We also updated how we computed our delta. Then we moved on to Project 5 where we take a look at how we are going to become more selective of our inputs. We took a look at our data and obviously the most common words appearing are also the most neutral words in the positive-to-negative ratios. Meaning that we are wasting compute time by processing words like the, it, they, he, she and due to the limitation of .split we are also processing characters like, , and .. Additionally, I forgot if it was project 4 or 5 but we also added a frequency counter, we just wanted words than appear more than a certain threshold. Finally, in Project 6 we came up with a synonym finder. I still have to learn about the math behind it but the gist is that we find the pos-neg ratios closest to the word we pick for example we want to find the closest synonyms for "excellent" and "terrible" and we find out where our cutoff level is. We then add this parameter to our system so that we can be selective of the words that we comb through and find out which words are closely related, i.e. finding the best context.

So there we have it. A sentiment analyzer from start to finish. This was a very interesting module. Theory intensive but still understandable. Also, it shows us how we can optimize our NNs to arrive to a conclusion faster which means faster value turnover to the business or the team we are supporting.

## Day 20: July 26, 2018

Sentiment analysis is done. The main goal of the lesson was to give us an idea on how we are able to frame a problem correctly. What this means is that we are able to figure out what the best course of action to take given our goals. We know what features to consider and what to discard based on the outcome that we want. Also, we are able to look under the hood of our neural network. We are able to simplify the operation in such a way that we do not do un-necessary computations or we can reduce the number of computations done to achieve a specific outcome. There was also an introduction to new hyperparameters that allow us to sift through the data and get only the most polarizing ones that will greatly affect the outcome and disregard the noise. Basically, improving our signal to noise ration so that we do not unnecessarily allocate processing to bloat words that have no bearing.

Target for today is Keras. For now, I have started the Keras course. Overview is that we are going to use Keras to do our neural networks from now on. Having finished the basics of doing NN via basic numpy, we can now move on to using Keras.

Finalizing my notes on the sentiment analysis projects. I am having trouble with Bokeh in the environment and I was not able to download it from home today, something to do with the internet connection/speed.

I can't wait for the weekend to come. Planning on taking on Keras and Sentiment analysis for my Project. For now, reading through on Keras and its functions and calls.

## Day 21: July 27, 2018

I have to slow my progress down today. Received increased load at work due to training commitments for the team. Will have to work slowly for now. I still have to my job to think about. Keras is still the target for today. First its important to learn about the Keras commands so for now that is the target. At the very least, finish this one over the weekend. I hate to push back objectives but I can't do anything about it. Anyways its just until November. :smiling_imp:

Anyway its now 4:41PM, I just finished Keras chapter 2. I was able to read through the Keras documentation and I now how some clue on what Keras is and I have a handle on how to do the most basic model which is sequential. They are right in saying that this is easier. In a way, I get it because the documentations are great and I am able to build on the previous topics about NN. For now I am going to watch the Mini-Project Intro just to get an idea of what I need to work on. I have to do some Lab works first but I doubt it will be done this night, I am still a single resource later. Hopefully nothing bad happens. Also, working on my resume today preparing for December.

## Day 22: July 28, 2018

Well, shift went south fast. So here we are, doing Keras at 8:00AM.  :thumbsup: The lab work right now is for Keras. An implementation of Student Admissions Analysis. This has already been done using the basic Numpy and Pandas, this time we are going to do it with Pandas and Keras. The prelab shows that this is a follow along lab, no wonder there was no TODO item in the lab notebook. Anyway, We would probably update Github for this one. This lab is more of a reinforcement learning of the Keras model.

So I am done with the first Lab: Student admissions. For now I am reading this [post](http://ruder.io/optimizing-gradient-descent/index.html#rmsprop) about Gradient Descent optimization algorithms. This post on [Visualizing and animating Algorithms](http://louistiao.me/notes/visualizing-and-animating-optimization-algorithms-with-matplotlib/) is actually awesome. Makes learning of the algorithms much faster. So what was basically introduced as SGD and MiniBatch GD are just the basics. In the post, I learned about Adam, AdaMax, Nadam,  NAG, RMSProp, AMSGrad. All math heavy but you get the point that they are improved versions of each other. Also, you get the idea that some should work more than others.
> __*There are no good and bad models, just right and wrong ones.*__

Just meant to say that every model, algorithm that we learn has its own use. Its about where we apply them to that counts. Imagine if you model a 2 input NAND gate with AMSGrad, that would be overkill when a simple SGD could suffice.

> __Incase you haven't noticed, its now on a more serious tone. Rarely placing emojis, and the tone of the logs are different. I guess it does consume you after a while. :muscle: push on!!!__

## Day 23: July 29, 2018

On to TensorFlow. Keras is basically, a wrapper for TensorFlow. Keras is good for creating a network quickly to proceed to testing and validation of the model. The problem with Keras is that it abstracts a huge part of the model. So we are now learning TensorFlow. TensorFlow is great for learning the operation of Neural Networks at a lower level.

A side note on Keras, there comes a time where in my experimentation I got a LOSS value of NAN after some epochs ~70. I think it might have something to do with the dimensions of my output on the first layer, 32. I think the model just fell off or the error function does not graph in a concave way i.e. the loss value went off track. This is worrying but it is good, it means I still have a lot to learn. :innocent: And because I have a lot to learn I have more dedication. Its a great feeling when you are so lost in your studies and that you have so much more to cover that you become more determined. Its a bit counter-intuitive but it works for me. It may get so daunting at times when you are so lost you have to be pulled out but there is that level of lost that feels like you want to do more, strive more and learn more. Or is it just me?  :smiling_imp:

So Tensorflow is quite heavy, its a new syntax altogether. This is going to spill over to tomorrow.

Just to get some perspective, I have another project due for August 20. 4 Weeks from now, and I have to cover 8 lessons to get there. This will be tough.

## Day 24: July 30, 2018

So I have started moving forward in Tensorflow intro for Neural Networks. The same time starting to go over the topics in Convolutional Neural Networks. I want to do this now so that we can proceed with building the project. There is a problem thought, there might be a need to use an external service from Amazon which is the EC2. This is a complication but it should be manageable. For now I am focusing more on getting towards the project phase. I know it would be long but it should be doable.

## Day 25: July 31, 2018

And just like that I am 25 days in on this pledge. I am actually happy about what I have accomplished so far. The learnings and the insights on what could be possible its great. I am excited to see how much further I can go on the next 25 days and what I can accomplish after 100 days.

For now I am focusing on the Convolutional Neural Networks topic and also doing something in parallel with the TensorFlow lesson in part 1. I am still wading through it. I slipped for some time last week and it has really been stressful lately but it should come back to normal now. __POWER OVERWHELMING__ is the key today. I was able to watch the intuition part of CNN from Intro to Deep Learning in Udemy. I am still finishing it but I found out that its good to have intuitions first on what is going to be accomplished and then take a deeper dive into the lessons in Udacity.

With regards to TensorFlow, I am having some issues with the requirements for the environment, for some reason it does not go through and some requirements are not found even though it is installed. The little things really are annoying sometimes. :fist: Right now, based on the Udacity progress bar I am 56% in to the TensorFlow lesson and I am 15% in through the Convolutional Neural Networks topic. The topics per se are quite small, 2 hours to 4 hours I think. Its the processing and the digestion for me that is proving some time. For now we shall continue on. Pushing through.

## Day 26: August 1, 2018

Finally able to finish the __Neural Networks__ lesson in the Nano-degree. I have to move forward and keep this pace. The next topics are going to be interesting. I am actually still doing the parallel play between the __Convolutional Neural Networks__ and the __Deep Neural Network in TensorFlow__ chapter. The reason is that I believe that TensorFlow is going to be used moving forward in the course. I'd much rather nail it down now and build on the knowledge today than go on with the course without an inkling of what is happening.

So update on this day is that I was able to finish fully my TensorFlow lesson. Learned how to do multilayer NN and also learned about a very important topic __creating checkpoints__. I emphasize more on _creating checkpoints_ because you are going to need save at some point especially if you are going to process a lot of epochs on a huge data.

## Day 27: August 2, 2018

I finally figure out how I can install opencv-python. This was the last package that I need to install to run the python notebook for the convolutional layer. In case I forget it, I have to run it via pip: __pip install opencv-python__. Additional note, it should be while __*inside*__ the environment. For now, I'll stick to the videos and note taking. Just a side note, the reason I can't download it is because the connection is  :shit: but at least I have the internet so to put things in perspective, I am still lucky :ok_hand:.

So I am now in CNN, and I have just finished 7 videos. Learned on how images are interpreted by computers: basically as a huge tensor with a base element of a pixel. Then we went on to discuss MLPs and how they are used for image classification and where they are trumped by CNNs: the explanation was that CNNs are more suited for multi-dimensional analysis where it looks for correlation not just in value but also in the relative position of the elements which obviously works well with images. Then we went on with Categorical Cross-Entropy for the loss function and how it is going to be used in the context of identifying an image: Basically, the model will output the probabilities of the labels and the error is taken from those probabilities taken together and compared with the probabilities of the one-hot encoded label. Then we moved on to validating models in Keras: there was an article about the MNIST data set and how it came to be and also about previous researches done on the data set and its results. Also was able to read more on the Keras documentation, I remember it was the __callback__ class where we get to store data of our training runs and see how our model is proceeding with its training. Based on the documentation on the __callback__ class there were also some interesting functions like _earlystopping_ which stops the training when the loss or accuracy is not improving and _adjusting LR on plateu_ where the LR is decreased automatically when a patience epoch threshold is met to ensure learning progresses.
<br>
That is all for now, will read more on this blog from [Machine Learning Mastery](https://machinelearningmastery.com/about/). I will find a way to download the opencv-python package later.

## Day 28: August 3, 2018

So, I was able to finally download the opencv-python package for the aind2 course. The plan now is to play around with the values in the network and go over the notebook to come up with the way the model was built. __:yum:__ <br>
Now the training and testing begins on the model. This is just Keras so nothing fancy, the objective here is to figure out where overfitting starts. __*Overfitting happens when the validation loss is higher than (by a significant ammount) the training loss*__. Here is an [interesting read](https://stackoverflow.com/questions/2976452/whats-is-the-difference-between-train-validation-and-test-set-in-neural-netwo) on the implications of validation sets on overfitting. We have known about data set spilts from our Introduction to Neural Networks, I believe its a Machine Learning concept or even an AI concept. But the idea is that we do not just burn through all our data in training, we have to have an idea of how well our model is able to predict an output or label from a data set that it has not yet seen before (think of it as the blind test). Depending on where you read, they say a good split is 20-test then 80-train, or 10-test and 90-train. The idea is that you want as much data as you can to train your model but have enough remaining data set to be able to test your model. This time we are adding another split to the __traininng data__ which is called the __*validation set*__. The validation set is usually 20% or so of your training data. The idea of the validation set is that it allows you to guage the tendency of your current model to overfit. Validation testing is done __while training__. In a way its like testing your data before hand, after each epoch or some epochs, to ensure that the increase in weights is actually going to contribute to the increase in accuracy of the model as a whole.
<br>
> When your training set increases its accuracy more and more after every pass but your validation tests are the same then the model is not actually learning anything new but simply memorizing the training set. This is a __sign of overfitting__.
<br>
When overfitting is detected the training should perform an early stop so that the model does not overfit the training data.
<br>
>Here is how I think of it: Let's say you are enrolled in a course. You run through all the possible materials of the course as dictated by the syllabus, this is your training. If we follow the basic train-test split, then after we go over through the materials multiple times we take a __FINAL EXAM__ which gives out the final grade for the course. If we follow the train-validate-test split, then after a going over the materials once we take a __QUIZ__ to check our learnings. We then take the result of the quiz and decide if we are ready to take the __FINAL EXAM__ or if we have to study some more.

## Day 29: August 4, 2018

Doing the Mini project today. THe idea is to figure out how overfitting can be avoided by different methods.

I just finished this tasks, and its now 8:34 PM. Did not feel well for the day. I will work on this some more. Target is to finish lesson from Local Connectivity on to Convolutional Layers then forward some more. I am aiming for the Quiz at #14. For now, its dinner then shower.

## Day 30: August 5, 2018

In case you are wondering, the results of the mini-project yesterday is on an actual physical notebook. I don't want to complicate things by creating a table in markdown. Anyway I was able to move past the mini-project and has started, belatedly, on my Convolutional neural networks notes.  :triumph: <br>The word convolution brings back university memories. Its one of the subjects I failed, Discrete Signals Processing. Terms like convolution, IFFT, Fast Fourier Transform, windowing.  :yum:<br>
For now I'll go over the explanation of what Convolution is and thankfully Udacity's animation does make it easier to digest, although I can't discount the fact that I already took this course twice I should be able to catch up by now.

## Day 31: August 6, 2018

This is a spill over of Day 30. I am the standby duty manager for Sunday going to Monday so its still my shift. Might as well do something fun about it. Anyway my dive into CNNs are now near the implementation part. My notebook already has topics covered from MLPs vs. CNNs to the more technical side of convolution like stride and padding. I even had the time to read and experiment on image kernels or filters. Learned __blur, identity, stobel was it or strobel?__. You can read about them also from this [post](http://setosa.io/ev/image-kernels/). And in the off chance that it might work, here is an animation of a convolution taken from __https://raw.githubusercontent.com/iamaaditya/iamaaditya.github.io/master/images/conv_arithmetic/full_padding_no_strides_transposed.gif__.

    <center>![Convolution](https://raw.githubusercontent.com/iamaaditya/iamaaditya.github.io/master/images/conv_arithmetic/full_padding_no_strides_transposed.gif)</center>
Te
For now I am already at the implementation of 2D convolution in Keras. Still reading through some of the arguments needed and there is a quiz. So I will be back shortly to update this post. For now, I'll push this to check if the animation would work. :smiling_imp:

I'm an idiot, I did not know that backticks != apostrophe. :poop: No wonder my code blocks were not working in both jupyter and in git. You never know what you learn.

## Day 32: August 7, 2018

Yesterday, I ended up in the pooling section of the lesson. I was more into the notes so I did not have any additional entries in this log. For now the target it to read and finish up on the pooling topic from Keras then move further into lesson. Yesterday was quite fun, learned how to compute for depth and shape of the convolutional layer (i.e. the wX) and learned about counting parameters as well.

I can't sleep. A daemon is telling me that I have to catch up and it keeps me up. I might as well read through and study the lectures. Found an interesting resource online, from the looks of it [this](http://cs231n.github.io/convolutional-networks/) might be a CS class. Looks interesting and skimming through it I am getting the idea of what a ConvNet (that is how I'll call them from now on ::). Then here is a resource, again from Udacity about [Loss functions](https://lossfunctions.tumblr.com/). Its like a Tumblr post about user submitted loss functions. Its quite fun to look at. Here is an image from that same link.
<center>![Loss Function Graphs](https://static.tumblr.com/c1127d32546080731a792febfa3dd631/xg196g9/S2Vnt8f21/tumblr_static_dp2izzbtsrsoc4cg8occ0o0cg_2048_v2.png)</center>

At the end of the day, I was able to read through the CS231n up until the ConvNet Layer. I am going to continue it tomorrow. For now I have a rough idea of what is happening. I am not gonna write about it here because I intend to add it to the notebook I made for Convolutional Neural Networks which will be uploaded once I finish the course.

## Day 33: August 8, 2018

Finally, Its now time to continue on to the readings. While on the way home I also watched the Udemy course by Kiril. I was watching the Intuition: Convolutional Networks portion. Again, getting all the information first and the theory then I will apply it in the Udacity projects. From the looks of it, Udacity is good for the __overview of the topic__ *but it does pay to dig a little deeper so that you grasp the technical/theoretical concept behind it.* At least that is how it works for me.

I also downloaded some datasets from Kaggle. One is about Fifa Man of the Match modeling I think. Another one was a data set for flower recognition and finally I browsed about the pictures for the Airbus Challenge. It was a good challenge advertised as a competition I think. The goal is to correctly identify ships from satellite images captured by Airbus. Its applications are really fascinating, I want to join but I have to be realistic. I need to finish this course first. :smiling_imp: Reading the Kernels in the competition is fun though. You get to see examples of how others are going about solving the problem.

 :satisfied:This reading material for cs231n is very informative. Its not overly theoretical, its just the right amount of insight to form a cohesive idea about the topics discussed.

>Don't be a __HERO__. When practicing the models, especially in CNN, try to find working ones that are already created instead of creating your own architecture from scratch. For now, especially since we are starting its better to have a working model from a built model that was tuned to our data set than a new model from scratch that would only work on our model. Having a template for this study is better that having a tailor fit suit.

I was about to start item 19 in Convolutional Neural Networks. It is a mini-project involving CIFAR. It looks like it will take a while when I download the dataset, ~ 3:47:15 was the ETA.:smiling_imp:. FML. No way. I have to actually use my AWS credits for this one. Anyway I have to sleep for now. Its already 12PM. I will get back on this soon.

How would you feel when the universe tells you __*"I got you"*__? I slept when I was not able to download the dataset. When I woke up, I checked my emails and there was this email from Kaggle. They are now offering free GPU enabled notebooks. :muscle:<br>
Don't you just love it when the confluence of things are in your favor? I am now done with the initial run of the mini-project. How about that? Big step in my progress today. I will do some more testing. I checked and with Kaggle's free GPU resource, I was able to run 100 epochs of the model at approximately 15 mins. Not bad.<br>

I am also uploading a copy of the notebook created in Kaggle together with this update in git.

## Day 34: August 9, 2018

Finally, some solid progress in terms of crunching through the lectures. __Having the right set of tools on hand while doing something really saves a lot of time and keeps you moving__. I hope that I can finish the course before the free GPU enabled notebooks of Kaggle runs out. I do have some credits in AWS but that is for later, for those compute intensive networks, or should it be used for CNN.  :open_mouth:<br><br>
For now I will continue on the Convolutional Neural Netorks Topic. This goes in parallel with the CNN in Tensorflow lesson. I say in parallel because training still takes a while to complete. Might as well read from other lessons to catch up. From the looks of the forum, I think I am behind by a huge ammount. I have until this weekend to finish all the pre-requisite topics before I touch the project. Must push on. :muscle:<br><br>
I ended at Transfer Learning in Lesson 2 and I am on Max Pooling Layers in Lesson 3.

I just finished ConvNet in TensorFlow. Actually, not fully. I am still about to do the read through for the MNIST code. I will run the code later. Also, I have to learn how to transition from importing the data set to simply reading it from a my disk in Kaggle. That is for later. For now I am focusing on CNNs in TensorFlow, specifically in going over the sample code.

I am viewing the code and it looks quite similar to ANN. Which makes sense considering that they should be based on the same concept but on a different operation. One critical thing I have to wrap my head around is the order of the arguments in the functions. TensorFlow already has built-in functions for neural nets including conv and pooling and we have taken on the weigths and varaibles in TF from the previous topic on ANN. This should take some time.

## Day 35: August 10, 2018

I did some studies last night. Mostly about initialization of the weights and how initializing the weights to ones or zeros are actually making the model train longer. The concept is about initializing the weights via the random generator.

I did some auditing of the remaining days for the project. Its only 10 days. Plenty of time relative to what can be done in a day but subject to the demands of the work. I still have to do my job before the course.

I actually was able to finish the Weights initialization course for this lesson. I am moving on to encoders tomorrow.

## Day 36: August 11, 2018

Doing autoencoders for today. I will be able to practice my tensorflow with this one. Was able to do the first auto-encoder, the simple forwards pass one. The idea behind an auto-endcoder is that it will compress your file and then decompress it in such a way that there is __minimal loss__ supposedly. By the looks of it, there is a noticable loss in terms between the input and output. Some are smudged while others are missing some tails or rounds.

The next thing to be covered is the encoder with CNN instead of the normal NN. First some [readings](https://distill.pub/2016/deconv-checkerboard/) on the deconvolution topic. This is going to be used when we do our upsampling in the decoding portion of our auto-encoder.

## Day 37: August 12, 2018

For today, finish auto-encoders via CNN. Final lesson before the project is transfer learning.

Okay, I do not have any output for today. But I did do something that I have not done in a while. __I SLEPT FOR THE DAY__.:smiling_imp: Also, I was reading on the prospects of the field of AI-ML-DL in ASEAN. I have this image from [this post](https://theaseanpost.com/article/prospect-ai-southeast-asia-0). The good thing is that there is a push to apply AI in several fields and ASEAN countries are joining in. The sad thing is that Philippines seems to be getting left behind, again. If the image is any indication, it looks like there is still room for improvement in the Philippine market. The question is who will push for the change. One interesting thing noted in the article was that even if there is an initiative towards moving to AI and automation, ASEAN investments are still small relative to the giants like US or China. One particular point provided was that the income class for most of ASEAN is mostly low. What this means is that the returns would also be lower or will take longer because nobody would be able to buy in to the services. Another interesting point is in the ethical debates concerning privacy. As we all know, AI-ML-DL requires user data and most of this is private or should at least have privacy. Its important to be able to learn about the current trends in the leaders in AI so that the region can better create laws to balance out the privacy and progress.

Then there is this article from ComputerWeekly about [how SEA keeps pace with AI](https://www.computerweekly.com/feature/How-Southeast-Asia-is-keeping-pace-with-AI). Again has a lot of important topics to consider. One interesting call out to the Philippines is the infrastructure for data, or lack thereof. To successfully scale any AI, there is a need for constant collection and transfer of data. While the country has kept close to the latest trends, investment in the infrastucture and network in the country as a whole is still lacking. Sure, the metro areas have relatively good infrastructure but how about those in the other provinces and areas?

 On another note, the openess of the country to data is high. The Philippines together with Singpaore is among the top in terms of open data implementation. But, relative to the global numbers, it can still improve.

 For moving in to the field, the article did state that __globally__ the manpower required for the AI field is still expected to grow. In terms of the growth potential, AI is still in its nascent stages. One point that could be taken is that:

>“AI/machine learning is simply not an easy technology to apply – it requires a good understanding of the business problem, and which data and machine learning tools will address that problem.”

Meaning that AI is, as it has always been, a tool that requires actual domain knowledge to be able to apply it correctly. Its not some sort of degree that gets you going. Its a tool that must be sharpened with experience and knowledge of the subject matter. Applying AI is dependent on the context of its field of application.

![ASEAN AI ADOPTION](https://www.theaseanpost.com/sites/default/files/10628_1.jpg)
Off topic: I have been browsing on Kaggle for possible project ideas and I found this: 

[Daily News for Stock Market Prediction](https://www.kaggle.com/aaron7sun/stocknews) in the data set. Interested to run through it and see what the data is like and find some applications. This one as well [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn).

## Day 38: August 13, 2018

I feel so refreshed. I slept close to normal hours for the first time in a while. Between working graveyard shifts, doing marathon sprints for projects and actually staying up due to standby duties, it has been __really draining__. Now at least I am a bit recharged. :poop:

So on to the quiz that was held up yesterday. Auto-encoders with CNN and tensorflow. I do have a physical notes for this section, just so that I can draw out the model and label which tensorflow function is needed as well as the arguments.

Placing this here:

```python
tf.nn.conv2d(
    input,  # The input matrix or tensor
    filter,  # Filter is not the filter size but the actual depth
    strides, # 
    padding,
    use_cudnn_on_gpu=True,
    data_format='NHWC',
    dilations=[1, 1, 1, 1],
    name=None
)
```

## Day 39: August 14, 2018

Actually finished the Autoencoder mini-project in Udacity. I had to run it in Kaggle so that I can make use of the GPU. I was not able to get the maximum accuracy. 

The plan for today is:
[x] Transfer learning start
[ ] Fashion Mnist CNN - Kaggle (open sourced)
[ ] Finish the Mckinsey report on AI in SEA
[x] Notes on the tensorflow functions

### Notes on TensorFlow CNN functions

`tf.Variable` - Using this for defining our _variables_ when initializing. We will need to call it as variable if we expect it to change at any time during our `Session`. Examples would be _weights_ and _bias_.<br>
`tf.placeholder` - Similar to `tf.Variable`, this is used to define _variables_. The difference is that the values defined as `tf.placeholder` will never change in the session. Examples for this would be _inputs_ and _labels_ or _targets_.<br>
`tf.layers.conv2d` - We use this to define our convolution layer. The key point to know for this is that _it will define our depth_. This would determine how many filters will be applied to our image.<br>
`tf.layers.max_pooling2d` - For this one we are applying pooling. The concpet here is that _it will change the **shape** of the input_. I believe it will decrease the shape. The output would be calculated depending on the strides as well as on the settings of the padding.<br>
`tf.image.resize_bilinear` - This will _increase the shape of the input_ to the new size in the arguments. The **new** pixel values will be interpolated depending on the method used, in this case it is bilinear. There are more image resizing options in the documentation. _**In other words, this is upsampling**_.
`logits` - This is not really a function in tensorflow that has only one call. This will vary depending on the architecture of our model. The general idea here, if we recall back to our Neural Networks lesson, is that this is $h$ (i.e. the __final output__ without the activation). We will use this later for the loss function.<br>__NOTE: This is not a rule that we call the output without the activation as logits, its just that in the context of tensorflow it is called logits.__<br>
`tf.sigmoid` - This is similar to `tf.nn.sigmoid` as they are aliases. Basically, it applies the sigmoid activation to our input.<br>
`tf.nn.sigmoid_cross_entropy_with_logits` - A built-in loss function in TensorFlow. There are various loss functions in tensorflow, it just so happened that we are using cross entropy for most of our exercises.<br>
`tf.reduce_mean` - This is a simple _mean_ function. This is done after our loss function since we want to get the average loss. It would usually appear as `tf.reduce_mean(loss)`.<br>
`tf.train.AdamOptimizer` - This is an optimizer call, in this case the Adam optimizer. There are many built-in optimizers in tensorflow as well. Just read the documentations on how to call them.<br>

##### Using the functions

Here is an important lesson to learn:

> We will not be able to use the tools unless we know what we are building.

What I want to say is that no matter what library we use, TensorFlow, Keras, SK-learn, it will still depend on how we were able to grasp the concepts. These tools are just there to make it easier to develop our models. Without knowing what to do and what to look for, these layers are worthless.

## Day 40: August 15, 2018

Plan for today is read the Mckinsey Report. Also, I am doing the MNIST Fashion dataset CNN in Kaggle. For some reason, I am having some issues on the data. Or I think I have a problem.

Also, to move forward, I am also watching the Transfer Learning course in Udacity. Transfer learning is a meta, like supervised learning. Basically, we use a pretrained model that others have trained and we will use that as our base for our new models. From the trained model, we add additional layers so that the results are going to get trained to our training sample (i.e. adjusting the model to fit our problem).

I am having some problems with the VGG code. Figured it out and it was due to the network I was on. The office network does not allow any connections to the github repo that was why I was getting an SSL error. I think.

So for now, I am going to try doing the MNIST Fashion and for some reason I cannot connect to the Notebook. I will have to do this at home. I'll open up the McKinsey report. If that still fails, I don't know anymore. It's turning out to be a crappy day.

I went home and opened up my Project - Dog Classifier. Here are some of the [resources](https://www.superdatascience.com/opencv-face-detection/) I scrounged for the task. I have to come up with a face detection algorithm. Also, I have to learn transfer learning as that will come in handy later in my project. 7 days to go. WOW.

## Day 41: August 16, 2018

Here is an interesting [Kaggle Kernel for Transfer Learning](https://www.kaggle.com/dansbecker/transfer-learning). Also, I finally figured out how to download and run my Transfer Learning notebook and its dependencies, VGG16 and tensorflow's flower dataset. 

Here is a note on [Transfer Learning from CS231n](http://cs231n.github.io/transfer-learning/) and here is one from [machine learning mastery](https://machinelearningmastery.com/transfer-learning-for-deep-learning/).

Okay, So I was able to progress with my Dog Classifier project. I have done the assesments part. I have a lot more to do though. For now, I need to do a network train the first instance of the netowkr. Its part of the Transfer Learning.

__To Do List:__

* [ ] CNN from Scratch - with explanation (1%)
* [ ] Transfer Learning CNN
* [ ] Fine tuning the transfer learning CNN
* [ ] Dog detector

So here are some of the notes I have with Transfer learning. We use transfer learning so that we can skip the training portion of the Conv layers. By making use of a _generaly trained model_, we can already have the weights and biases that will work for a generic task. We are to use the pretrained Conv Nets to either _be the start of our model (i.e. we will still adjust the weights) or we can add to it our more task-specific feature extractor_.

If we are going to use a pre-trained model as a fixed-feature extractor we simply have to change the FC layers and keep the original layers fixed. What we are doing is changing how the model decides which is important and which is not only on the Fully Connected layers, we just use the Conv Layers as a base model.

We can also use a pre-trained model as the starting point of our _fine-tuned model_. In this scenario, we will adjust __all the weights and biases__ of our network via back propagation. From the FCs to the first Conv Layer from the input. This way we have a pre-trained model and we cutomized the weights even further so that it is specific to our task. One thing to note when doing a fine-tuning of a pre-trained model is that it is actually not a bad idea to lock the first few layers of the Conv Layers. The reason behind this is that the first few layers should only come up with generic features. Training these layers would often cause more issues with overfitting later on.

Research Topics: Zhima Credit of Alibaba.\
Nathan McAlone, “Why Netflix thinks its personalized recommendation engine is worth
$1 billion per year”, Business Insider, June 14, 2016.

## Day 42: August 17, 2018

Progress done on the project. I was able to do the CNN from scratch. Next thing to do is the CNN from Tensorflow, I have to study it first. There is a pattern here. I will learn it. I have also done some readings on the McKinsey report.

For now I intend to at least start the write up on the VGG-19 transfer learning. From what I have read yesterday (see the logs above), there are 2 ways to do transfer learning. One is to keep the conv-pool and edit the FC layers. The next one is start of with the model and keep the weights there. I think that is what the project wants to happen. Which shoud be doable. I am unable to run this on Kaggle for some reason which is sad. I am also having some issues with the connection to the workspace. It resets sometimes with some edits lost. For now that is it. I'll add some more later.

## Day 43: August 18, 2018

Today, the plan is to figure out the VGG 19 model. possible source of pre-trained models for download [VGG-19](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz), [ResNet-50](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz), [Inception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz), [Xception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz). Do note that these are pretrained models and will therefore consume space. For example, VGG-19 is ~800MB in size.

For now I have to write the code for steps 5-7 of my project so that I can initially submit and get feedback. I also have to defend my choices for the architecture so good luck with that. That would mean reading some more VGG19 notes to get the concept of what happened inside the VGG.

## Day 44: August 19, 2018

Went out with friends yesterday. Did not do much about the project. So there is no additional progress to reported for yesterday. For today I am on Standby again. I will have to do some smoketesting for my job :poop:. I am going to read about the VGG19 lectures for today. The goal is to be able to explain and come up with the model.

For now, I will code in VS code and try to run it later in Udacity. I will also have to read the VGG or inception notes to provide a backing on the choices. Also, I will be reading Mason's book. Intending to finish it now.

Here is another [reading material](https://adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html) which contains the papers that is interesting for Deep Learning. For now my priority commintments to the standby shift is completed. I will nap for a bit and then continue with this coding. I am thinking of changing my approach to the inception model instead of VGG-19. Reason for this is that I am intending to add just the fully connected layers and Inception already has a good accuracy for feature detection. This would be fun :smiling_imp:.

Okay, so I researched a bit about the Inception network. Its Google's submission to ImageNet 2014 competition. One of the ideas behind the architecture is that it is structured in a way that the system is designed to actually fit a computational budget which was imposed for the practicality of the network to be used in other applications like embedded or mobile apps. Inception is geared more towards efficiency which we greatly want considering the application we want has limited computational power (relatively).

Had great fun and insights reading the first chapter of the GoogleLeNet (Inception) paper. Wow. They were able to explain the trend as well as the drawbacks to those trends in creating Deep Neural Networks. They also made it a point to explain why there is a need to be efficient in creating the network instead of just blindly enlarging it in terms of width and depth.

And an application went unavailable. I cannot catch a break. __SERIOUSLY__. :poop: Its now 7:43. Almost 14 hours awake. And only a few of those are spent for the project. Mostly reading. This is really disheartening.

## Day 45: August 20, 2018

    Why InceptionV3? Well for one thing, I was actually impressed by the point made on the paper "Going Deeper with Convolutions" describing the idea behind the Inception network. For one thing, it interested me because they mentioned that this is a model that they designed with a computational limit in mind and geared towards mobile or embedded computation which means that efficiency for the model is key as opposed to highest accuracy. Given that the model we are going to do does not really require that much of an accuracy. Also, in terms of base accuracy, if I read it correctly between VGG, Resnet and Inception models, Inception models have the highest accuracy in its basic vanilla version so that always helps.

Found out another interesting fact about inception: __It processes visual information at different scales and aggregates the results to figure out the features of the image.__ _How cool is that?_ That is like an HDR image trying to come up with a sharper image only in this context it is trying to come up with a good feature.

>Given relatively large depth of the network, the ability to propagate gradients back through all the layers in an effective manner was a concern. The strong performance of shallower networks on this task suggests that __the features produced by the layers in the middle of the network should be very discriminative.__ By adding auxiliary classifiers connected to these intermediate layers, discrimination in the lower stages in the classifier was expected. This was thought to combat the vanishing gradient problem while providing regularization. - Going Deeper Paper

The statement above, forked from the paper, is another nugget of information. It change how I though of the network architecture generally. The idea that it is the middle layers that actually improve the ease of backpropagation. To achieve this they added more convolutional layers on top of the inception's layers that are now weighted. The way I understood it is that they sort of added a capacitor in the later stages that will allow the voltage not to sag so much in value. Only in this context, its the gradient that we do not want to sag so much so as to prevent the vanishing gradient issue.

Man, I am all over the place. Devouring labs and papers and everything else I can get. Just learned about Bottlenecks. Its not really slowing the network down. Its just named that way due to the way it looks if you imagine a typical network. __Bottlenecks is the layer we have just prior to the classification layer done by the output layer.__ By these defenitions, its easier to see that the top of the model resembles a bottleneck slowly trying to taper of until only the number of classiications are left.

One possible improvement I can think of while reading some materials to decide on this architecture is the possiblity of regional bounding and labeling, I think that's the concept behind it. Basically, it will look for points of interest or fields of interest in an object and do a convolution on regional fields instead of doing the entire picture. I think that is a great idea since we are more interested in finding subjects in the image and classifying that subject instead of blindly applying convolution on "blank" spaces or empty fields in the image trying to find features. Its quite interesting to now that the idea is already there. I would really want to try it out later.

Officialy found the [TensorFlow Mobile iOS lab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2-ios/#0) manual. THere is also an entry for Android and android mobile development for TF. Interesting.

For now, I filed for a Sick Leave. I have to sleep. This draft is due in 3 days. I have a plan already. First: Review the Transfer Learning Notebook again. Second: Get the context and transfer it to the current project. Third: Run the code and check that it does meet with the requirements for the review.

I have to figure out how to do the Optional Part as well for fun.

## Day 46: August 21, 2018

So, 2:00 AM and running through this project. I have added a GAP layer to the pretrained models.

While training the models, I was reading throgh some post on LearnOpencv.com. I was reading about [Fine tuning on Keras](https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/) and [Transfer Learning on Keras](https://www.learnopencv.com/keras-tutorial-transfer-learning-using-pre-trained-models/). I read somewhere that transfer learning is when you use the pre-trained model as the classifier and add your own top layers while fine tuning is when you make use of the pretrained models as the starting point of your training. This does make sense. Onwards.

Back to the coding of the TF for Inception. Checking on the output of the inception model via `print(train_inceptionV3.shape[:])`, I noticed that the shape of the output is (6680, 5,5,2048). This would make the input to our layers as (5,5,2048). Reading from a post by instructor Alexis Cook about [GAP layers for object localization](https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/), there was an argument there that the GAP layers are being used to reduce the total number of parameters which in turn prevents overfitting (and by extension less computations).

Then I added two simple Dense layer before the final Dense layer for the output. I was about to add a 500 unit layer before the final output but the summary showed that there would be more than 1,000,000 parameters in total. That would not be good. So instead of going wide I went deeper. I know that spillting a 500 unit dense layer to two 250-unit layers are not the same (width vs. depth) but I want to ensure that there is another decision maker redundancy before the final layer. Doing the deeper route led to 608,383 total parameters.

Finally, I am able to get it to work close to the intended purpose. I have yet to train my FC layers to more epochs, I think it would benefit more on the decision making FCs. I'll add an early stopping just in case and set the epochs to 2000 with patience of 5. :muscle:

Right now I am having issues with the Workspace again. As long as the progress (codes) are saved then we will not have a problem :smiling_imp: While waiting for the model to train here is an article about [9 things to know about TensorFlow](https://hackernoon.com/9-things-you-should-know-about-tensorflow-9cf0a05e4995).

Made my first submission for the CNN Dog Breed classifier. I heard back from the review team. The initial submission requires minimal re-work. Most of it is in answering the questions which I though was optional. :poop: Turned out its the coding portion that is optional. Not answering the questions. I also raised the issue about the 'dog_names' being problematic in a sense that it is still displaying the index. They said try to restart Jupyter and run all the codes initially. Also, they suggested that I remove all the duplications in the code cells I made below to avoid further errors and to limit the points of failure.

## Day 47: August 22, 2018

Changes that needs to be done on the notebook:

- [X] Fix the 'dog_names' issue (if possible)
- [X] Answer Question 2: Face Detection
- [X] Answer the question on Build from Scratch
- [X] Answer the question on the difference between build from scratch and TF
- [X] Fix the images used in testing. Remove the ones from the original repo.
- [X] Switch the face detection and dog detection in the algorithm

I have a lot of things to do for today with regards to the project. Nothing major really. While waiting for the review, I continued on CNN with the lesson from Sebastian Thrun about Melanoma detection using CNN. Its an interesting topic and it does have implications on the society.

I also read about OpenCV and face detection to answer the question posted in the project. Here is one article refered to by the reviewer about [OpenCV for face detection](https://memememememememe.me/post/training-haar-cascades/). Here is another one about [Face detection using HAAR cascade classifiers](http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php). Finally the last referal article is [OpenCV's documentation on Haar Cascade classifiers](http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php).

From what I understand on the Haar Cascade face detector, it actually is limited to a few unique features when detecting a face. It will detect features based on an XML file for example `harcascade_frontal_face_alt.xml` or `haarcascacde_eye_tree_eyeglasses.xml`. These XML files will dicate what features can be detected. The limitation comes, as already presented in the context of the question, when there are slight deviations on the input images that the detector cannot accomodate.

In line with the topic of this project, I think CNN would be a great alternative for finding faces in a picture. Instead of the detectors being a fixed file pre-loaded in the xml, with Deep Learning and CNN we can possibly accomodate more "unique" inputs. For example a partial image or an image that is rotated.

I initially thought that Haar Cascade was transfer learning. In a way, it seems to be transfer learning (i.e. it does train the detectors to provide an xml file). It just so happens that Haar Cascade is not that great in accomodating deviations in the input and will have a problem with scaling to a larger dataset. With a larger data set then I believe a solution involving CNN would be more appropriate.

[Batch Normalization in Fast.AI](http://course.fast.ai/lessons/lesson5.html) . [Used in CCN](https://www.kaggle.com/kentaroyoshioka47/cnn-with-batchnormalization-in-keras-94) acheived 94% accuracy. From what I understand, you will use normalization to increase the speed of the training by making sure that all the inputs are centered. In a way it reduces co-variance shift. If you think about it, what it means is that we try to keep our inputs "normal" by setting it to play around a defined range (usually set by Standard Deviation). This will cut-off the outliers which will take some points of our accuracy but the speed increase would be beneficial.

```python
### Here is a sample code for using normalization. It is done before the activation.
model.add(Conv2D(64, kernel_size=(filter_pixel, filter_pixel), activation='relu',border_mode="same"))#1
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(droprate))
```

It is almost 6:00 AM and my shift is almost ending. Finally submitted my codes for review. I'll sleep and see what happens.

Actually, did not sleep *YET*. I went on and learned about Cancer detection AI by Sebastian Thrun. It is used to detect Melanoma in its early stages when it is not yet that fatal. Learned about ROC (no not region of covergence). Almost done with it.

Its 1:00 PM and I opened up Gmail to see that the project submission has been reviewed. __It meets all specifications__. So happy that I was able to finish the project on CNN. So pumped. Its just day 47 and looking at my progress really makes me happy. On to the next topic: __Recurrent Neural Networks__.

Also, I made a promise that I will help out a friend regarding Kaggle. So while I am working on my __RNN__ intuitions I am also going to do a collaborative project.



That is all for later. For now, I am gonna play __Yuri's Revenge__ and I'll finish The Subtle Art of Not Giving a Fuck. I am 40 pages short of completing the book.

## Day 48: August 23, 2018

Playing around with the concept of Datasets in Kaggle. For some reason I cannot get the CSV file to work when I upload it.

In the meantime I had fun reading [Elite Data Science: Becoming a Data Scientist](https://elitedatascience.com/become-a-data-scientist). Also, I was able to start a new Kernel for a collaborative side-project although we would need to discuss first until where the project would lead to. The idea is to create a genre labeler from the IMDB dataset. The Kernel can be found [here](https://www.kaggle.com/iocfinc/genre-tag-imdb). We are still in the early stages of it, mostly data clean up. __It is great fun though__. Just today I figured out how to prepare the dataset (which I did not use by the way due to confidentiality). I was able to do a simple _Seaborn_ bar plot of the common words. I was also able to learn how to create a DataFrame from a counter using Pandas as well as create a New DataFrame. __I am really enjoying this.__

In terms of the Udacity course, I am finished with the Cancer Detection AI of Sebastian. When I arrive at the office I can proceed with the RNN.

> One of the key point we have to know is that AI in itself would be disruptive in the way society would work and it already is disrupting most of the work now. The power of AI to do tasks that are repetitive would give, those that are willing to use and explore it, the power to leverage impact and scope of work. One interesting example given was how farmers in the US before can feed 4 persons. Right now, with the proper application of AI, it can feed up to 155 people. That is an almost exponential increase in productivity.

## Day 49: August 24, 2018

Finally started browsing through recurrent neural network. The project which is a TV script generator is due in 20 days. So day 70. By day 60, I am expecting that all the lessons and mini-projects should already be downloaded.

I was working on the Genre Tagger for IMDB in Kaggle. Still on data pre-processing. I now have a new table with words and genre as the indices. I am planning on cleaning up data some more. I think I need to cut off some of the non-polarizing words, (i.e. a, the, for, it). In terms of Recurrent Neural Networks, I was able to browse through the readings. Its quite long, 4 hours worth of watching and intuition lectures. Then some projects and quizes.

For now, back to the genre tagger. I have until tomorrow to work on this. I need to create a histogram or distribution list at least of the words.

## Day 50: August 25, 2018

Its now halfway through this pledge. So happy about my progress. Still a long way to go. To 50 more days. I am now on Recurrent Neural Networks. Neural Networks with memory. For now, in the Nanodegree its more on lectures and theories of how RNN works. In terms of project, I am working on the Kaggle Kernel for IMDB genre tag. More projects and more applications to build. This is so much fun. In terms of other things, I am done with __The Subtle Art of Not Giving a Fuck__. Up next would be finishing __Thinking Fast and Slow__.

Still have a lot to do. I am looking forward to applying my learnings. Found this opening in __Open AI__ for a __[Machine Learning Engineer/Researcher](https://jobs.lever.co/openai/588c1d80-4632-4d5c-a535-9f2c8c80c501)__. Based on the listing, one of the characteristics they are looking for is novelty and . This is [the challenge](http://app.getpy.com/start#hNAQTq5e) that was posted on their site.

## Day 51: August 26, 2018

Will finish Recurrent Neural Networks intuition portion for today. I have 9 more days to finish it. I also have been reading requirements for OpenAI researcher and Engineer. The plan is in line with the move to AI field. I am basing it on the email I got from EliteDataScience. It outlined how to start [building your own resume](https://elitedatascience.com/resume-tips). Also, here is another article from [Codementor regarding portfolios](https://www.codementor.io/mgalarny/how-to-build-a-data-science-portfolio-mcnz7sxlt?utm_content=posts&utm_source=sendgrid&utm_medium=email&utm_term=post-mcnz7sxlt&utm_campaign=newsletter20180822).

I am also reading on Pandas Documentation. Specifically how to [handle dataframes like in SQL](http://pandas.pydata.org/pandas-docs/stable/comparison_with_sql.html). With regards to the intuition portion of RNN, it seems like they again made a review of feed forward and backpropagation topic. This makes sense because RNNs were created to address the problem of _Vanishing Gradient_ that is affecting the original neural networks.

> One of the major difference with a feed forward neural network (FFNN) from a recurrent neural network (RNN) is the absence of __sequences__ and __memory__.

__Sequences__ are now the input of our network. For example a sequence of words or the 5 previous closing price of a stock. Then we have __memory__, this is simply the output of hidden layer neurons being stored and fed back into the system for the next iteration. This structure is defined as the __Elman Network__.

<center>
![Elman Network](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Elman_srnn.png/330px-Elman_srnn.png)
</center>

There are two models that are often used when dealing with RNNs. One is the __folded method__ where the _state/memory_ is seen as looping back to the input. The other one is the __unfolded__ where the _state/memory_ is fed to another model but at a future time (i.e. the initial state __s__ is fed to the input at t+1, s at t+1 is fed to the input at t+2 and so on).

![Folded model](https://d17h27t6h515a5.cloudfront.net/topher/2017/November/5a1c955f_screen-shot-2017-11-27-at-2.44.11-pm/screen-shot-2017-11-27-at-2.44.11-pm.png)source: Udacity DLND

![Unfolded Model](https://d17h27t6h515a5.cloudfront.net/topher/2017/November/5a1ca463_screen-shot-2017-11-27-at-3.48.31-pm/screen-shot-2017-11-27-at-3.48.31-pm.png)source: Udacity DLND

One example of RNN is when we want to detect a word, for example `hello`. To do this we input individual letters and the system figures out the error by decreasing the error every time the correct sequence is detected until eventually the correct word is detected. For example `h` has an error or .9 then `he` will have 0.8 and `hel` will have 0.7 until we reach `hello` in which case the error would have been ideally 0. Do note that this is just an example. Training the RNN involves backpropagation through time.

In __Back Propagation Through Time__ we are going over not just the weights and the input at the _present time_ but also consider the __state__ of all the previous times. The Gradient descent is therefore not just on the current time forward and back but also the preceeding states in time. I know, it sounds like a weird Back to the Future concept but it does explain how RNN approached the vanishing gradient problem. This time __the contribution of *ALL* the preceeding weights are considered in the computation and adjustment_. Obviously the farther back in time the state was the farther the chain rule has to go back and the gradient would still vanish but by accumulating all the gradients to adjust the weight instead of just one we can have a bigger value for the gradient. Its in simple terms a _Recency Bias_, although that is now how the is used. What I am trying to say is that the most recent inputs and state will, obviously, contribute the most to the adjustment of the weights.

![RNN Output](http://quicklatex.com/cache3/a1/ql_4cd5f21dec1523e6957fe9491ca9c1a1_l3.png)

So I sort of figured out how to do BPTT. The idea is that we need to be sure with respect to what are we adjusting the weights.

![Sample Backprop - 1](https://d17h27t6h515a5.cloudfront.net/topher/2017/December/5a24fe85_screen-shot-2017-12-03-at-11.34.41-pm/screen-shot-2017-12-03-at-11.34.41-pm.png)source: Udacity DLND

> Quiz Question
>Lets look at the same folded model again (displayed above). Assume that the error is noted by the symbol E. What is the update rule of weight matrix U at time t+1 (over 2 timesteps) ? Hint: Use the unfolded model for a better visualization.

I'll try to add the images but if they will be removed then I have no complaints there. For this one we need to get the update rule for weight matrix at U for 2 timesteps (t and t+1). So from here we get the idea that we need to get from $\bar{y}$ to $U$ for two timesteps. As stated in the hint, we are better off using the unfolded version of it. In the image below we already have the first path from $\bar{y}$ down to $U$ so this would be for time t+1 (i.e. the present).  This is the first path.

![Sample Backprop - 2](https://d17h27t6h515a5.cloudfront.net/topher/2017/December/5a259f26_screen-shot-2017-12-04-at-11.16.19-am/screen-shot-2017-12-04-at-11.16.19-am.png)source: Udacity DLND

We also have to consider the path to $U$ for time $t+1$. In this case there are _two paths possible_ between $\bar{y}$ to $U$ at $\bar{x}-sub{t}$ (referring to the image). One is to take the path from $Z$ node to the time $t$ and the other is from $S$ node to time $t$. The path is more clear in the succeeding images.

![Sample Backprop - 3](https://d17h27t6h515a5.cloudfront.net/topher/2017/December/5a25a02b_screen-shot-2017-12-04-at-11.14.30-am/screen-shot-2017-12-04-at-11.14.30-am.png)source: Udacity DLND

![Sample Backprop - 4](https://d17h27t6h515a5.cloudfront.net/topher/2017/December/5a25a091_screen-shot-2017-12-04-at-11.12.31-am/screen-shot-2017-12-04-at-11.12.31-am.png)source: Udacity DLND

Once the paths have been made, its now a simple application of chain rule. Come to thing of it, it looks like a discrete math question: Find how many paths from $\bar{y}$ to $U$. The answer for the question would be:

![Sample Backprop - 5](https://d17h27t6h515a5.cloudfront.net/topher/2017/December/5a25a757_screen-shot-2017-12-04-at-11.48.22-am/screen-shot-2017-12-04-at-11.48.22-am.png)source: Udacity DLND

Note: This has always been a problem for me, -t is past for me and +t is future. But that is considering where your point of reference is. From the image, the present time is already t+1 so we will use that. A possible alternate would have been the present is t and the last sequence was the past (i.e. t-1).

The formula below is the most basic representation of an RNN.

![RNN-Formula](https://d17h27t6h515a5.cloudfront.net/topher/2017/November/5a04ea8c_screen-shot-2017-11-09-at-3.53.12-pm/screen-shot-2017-11-09-at-3.53.12-pm.png)source: Udacity DLND

Final note on RNN. With the addition of the previous states to the equation of getting the output state we were able to accumulate the gradients of the previous state and get a bigger gradient value. This however would still vanish if we backpropagate for more than ~10 steps. This is inherent to the mathematics involved in getting the gradient. The temporal dependencies will always decrease geometrically. To avoid this, __Long Short-Term Memory (LSTM)__ was created to address the decay for RNNs. One problem that the RNN is known to have is the __exploding gradient__ problem where our gradient actually grows uncontrollably (i.e. we will diverge instead of converege in our error). To solve the exploding gradients we use __gradient clipping__.

## Day 52: August 27, 2018

Interested in listening in to this [Podcast](https://80000hours.org/podcast/episodes/the-world-needs-ai-researchers-heres-how-to-become-one/). Also, Ate wants me to try her netflix account. Temptations. :poop: For today I will do some chores first and read some book. Then later I will continue on to LTSM topic by Luis. It might have some lectures and reading to go through.

Working again on the Kaggle IMDB tagger. Still cleaning up the data. This is actually taking up a while to complete. I was able to fix the data. I was able to create a new `Genre` called `Others` which was the comibination of all genres with less than 100 items. Then I was able to make a new table with 100 entries for each genre. Come to think of if, it might still be a small dataset to do NN work on. assuming we have 20 test and 10 validation we have 70 train sets for each. Unless I am thinking at it wrong. In any case, it has been paused for now. I encountered an issue when I tried running my word counter. Will fix it probably tomorrow or next weekend. For now the focus is on LSTMs and RNN.

## Day 53: August 28, 2018

What is LSTM? __Long Short Term Memory__, which is used to retain memory and help solve the vanishing gradient problem encountered in RNNs. From what I can get, the concept is that we use memory to retain context. For example, we want to correctly identify an image. The previous images has been a bear and an owl. When our system sees the image it is deciding between a dog and a wolf based on its features. But knowing that the previous images has been in the context of animals from the forest, the system would choose to classify it as a wolf instead of a dog.

LSTM works by keeping both long and short term memories as inputs and then using these memories to come up with a prediction as well as the next long and short term memory. To do this, the LSTM has multiple gates (4 to be exact) that it uses to keep track of long term and short term memory, the prediction and the new entries for the long and short term memory.

__Learn Gate__ is used for the short term memory. Its input is the previous short term memory and the latest event input. It would then pass through the ignore logic where the older items in the memmory are forgotten. The rate at which the previous items in the short term memory is ignored is decided by the _ignore factor_.

__Forget Gate__ is used for the long term memory. Its input would be the previous long term memory (LTM of t-1) which is hen multiplied by a _forget factor_ which is a function of the previous short term memory, the current event which is evaluated with the forget weights and activated by sigmoid.

__Remember Gate__ is used to get the new long term memory that will be used by the next timestep. Basically it is the addition of the previous long term memory and the previous short term memory. These two will then add up to make the new long term memory.

__Use Gate__ is used to get the new short term memory. Its input is again the previous long term memory and short term memory. This time, instead of mathematically combining the two it is the evaluation of the parameters to come up with an updated short term memory.

For more details on LSTM make use of [this link](https://skymind.ai/wiki/lstm#long). I have finished reading up on LSTM topic. the articale in skymind.ai which is linked did a great job in explaining more how the LSTM gates are used in a grander scheme of things. The simile on life is also a big plus. I am now on to application of LSTM by Matt which I will do in the office later. For now, I am fixing the Kaggle kernel.

Update on the Kernel, its now closer to moving to neural networks. Here was one [Kernel for Data Science beginneres](https://www.kaggle.com/glsahcann/data-science-for-beginner) which sort of helped me in this sprint. Also, __Stackoverflow__ as always. I was able to finish up cleaning the data (removed the punctuations via `maketrans` and `translate`). I then fixed the `others` genre for genres not reaching the 100 min count I set. The heatmap is also fixed for now.

I think I can start moving up to doing the NN. First step later would be to one-hot encode the `genre` column. Then split the data into _test, validate and train_ sets. That is all for later. Also have to get progress on my _secure the human training_.

> Update on the timings 8 more days to finish 4 more lessons. Keep at it. :muscle:

__:poop:__

## Day 54: August 29, 2018

I was able to finish my _Secure the human training_ for work commitments. For today, I have to start on LSTM/RNN applications. Later would be to transfer the revisions to Kaggle and commit it as another version. Focus now more on Udacity's nanodegree.

RNN and LTSM implementation. The mini-project is to create an RNN that will output a "semi-comprehensible" text based on character inputs. I was about to go over it but I was drawn into [this post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) by Andrej Karpathy regarding RNNs. He also has a post [regarding Neural Nets](http://karpathy.github.io/neuralnets/). The Karpathy post is quite long and I am unable to read through them. Had to stop and go to work for now.

## Day 55: August 30, 2018

Still stuck at RNN implementation. I cannot seem to move. Everytime I try to open the classroom I get stymied. Procrastination at its simplest form. :poop:. We shall see what progress can be done today. I might have to reformat my laptop for work. Also, I had to setup my AWS account for deep learning. I did the registration and redeemed my credits. I have requested the increase in my instance limit. I'll see when the approval gets in.

While waiting for the IT office to open up (bit of context: I work nights and they usually get in around 8-9AM) I was watching some videos on Youtube. Like I said, procrastinating. I came across this [video](https://www.youtube.com/watch?v=UuAJMzpoq5E&index=167&list=WL) about DJ Patel, chief data scientist of the White House. I just found it interesting what he mentioned were traits that Data Scientist should have. Curiousity, Diversity and teamwork as well as solving local problems and studing data.gov datasets that the US government has already released.

Another procrastination [chore](https://www.16personalities.com/intj-personality). :poop:

## Day 56: August 31, 2018

This is going to be another slow progress day. I am backing up my laptop files. I have to reimage to Win10. Will do Kaggle and watch UD lessons later.

Its now 14:42. I have just finished most of the required installations for the migration. I was not able to do any other thing related to the degree today. Actually, I was able to do something in Kaggle but I again got stuck. I already had the heatmap in check but I am having trouble doing the Neural Network part of it. Also, the code has not yet been commited. I still have to work on the draft. For some reason the changes I made (which contained the fix of what I am stuck right now) was not saved. I think it is my fault. I should have downloaded the version. Or do I need to upload a version?

Studied [this post](https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/) for Pandas data manipulation. `loc` and `iloc` and `ix` calls. I need it to do the Kaggle network.

I will be continuing on the RNN at udacity today. This would be a productive day. I am not on stand by either so that is great.  More time to focus here.

## Day 57: September 1, 2018

Its finally Ber months. Its now close to December. The pressure is now on to finish this one and work on my projects so that I can build my portfolio.

Today, I am finishing RNN Implementation lesson. I will go over the exercises in the notebook and I'll try to squeeze the Hyperparameters lesson as well. The createion of an LSTM for Tensorflow is quite odd. Odd in a way that it is not similar to a basic CNN or NN for that matter. It has its own quirks. Quite big ones actually. I am having some trouble wrapping my head around it for the moment. But must move on. Must keep going forward.

In terms of the Kaggle kernel, I think I have a solution to the problem I was having. Basically, create a set for the words that are split from the row. After creating the set, we will need to change the value of the columns with the column name in the set of words to 1 from 0. I do not have the time right now to do this as Udacity takes precedence.

Another possible solution would be to reuse what was made in the sentiment analyzer project to this kernel. Again, I'll try to do a sprint for this tomorrow. For now Udacity ND takes priority.:poop:

Also I need to install git bash again, SAD.

## Day 58: September 2, 2018

8 more days to the deadline. Done with Hyperparameters lesson for today. A complete review of the Hyperparamenters from the earlier lessons in neural networks. Some of the new insights taken were that there are 2 types of hyperparameters: One set is for the optimization and training like normalization, the second set is for the speed like mini-batches etc. Smaller batches lead to slower training time. Bigger batches tend to oscillate. This is similar in relation to learning rate. Speaking of learning rate, if you read the documentation in tensorflow, there is a setting on learning rate decay. What this means is that the learning rate will decrease at a fixed rate after a fixed epochs. Then there is also a training method for early stopping. This would allow us to stop the training phase after a few "patience" epochs are reached without the error decreasing. This saves us time by ensuring that we do not unnecessarily train the network when it is not improving.

Also there is a research regarding depth of the layers. 3 layers outperforms 2 layers but there is no significant difference the more layers you add. The obvious extension would always going to be the convolutional neural network.

Also, look at Elon's resume below. Talk about goals. Also, been browsing on this [article](https://www.greatlearning.in/blog/5-must-haves-on-your-artificial-intelligence-resume/) about what to put in your resume for AI-ML-DL track. I still have a long way to go but now at least I know where to go and get them.

![GOALS](https://images.news18.com/ibnlive/uploads/2018/03/elon-musk-one-page-resume.jpg)

## Day 59: September 3, 2018

Plan for today. Coding and dissection of the three mini-projects `intro-to-rnns`, `embeddings`, `sentiment-rnns`. I was able to run my `intro-to-rnns` earlier although it took a while to train. It showed a good ammount of progress though. I timed it, almost 30 mins for the entire pass. For now I have to dissect it to undestand what actually goes on and view the documentations on what could be improved. Then I have to go to the office for the upgrade. This is going to be costly and time consuming. 

## Day 60: September 4, 2018

40 more days to got. Power through. I browsed through on the next topics: GAN and Reinforcement Learning. There are intense lessons ahead. Some of the lectures are worth 10 hours for GAN. For RL, the topics are not that long BUT the quantity is also huge because it is mostly situational so its almost the same. For now I am finishing the RNNs topic. I am actually dissecting how the RNN was made in the characterwise RNNs. Then after that I am going to dive through how it is being made as word2vec or embedding.

I think one of the more important things to consider when trying to understand RNN applications is that there are actually 4 types of architecture being used. This is based on the relationship between input and output of the architecture. There is __one-to-one__, where there are mutliple inputs and multiple outputs (for example the characterwise RNN). There is __one-to-many__ which is for script generation (I think, where you have one key sentiment and then it generates base on that). Then there is __many-to-many__ and then __many-to-one__. For _many-to-one_ and example could be a sentiment analyzer where you take in a stream of words and come up with the context. I think I am confused between _many-to-many_ and _one-to-one_.

## Day 61: September 5, 2018

I am going over the RNN dissection. I think there is an issue with the matching of the solution notebook to the actual problem notebook. I am having issues with the `tf.contrib.rnn.BasicLSTMCell`. I am slowly getting over the requirements. I have to start the project later in the workspace. I need to start doing the codes. Less than a week to go.

I'll read TensorFlow documentations regarding LSTM later. I need to figure out how the cell is to be created. And the connection is crap for today. I can't even download extensions. What is with that? Seriously?

I also have not updated the log files in the repo. Git bash is still not fixed.

```python

tf.reset_default_graph()

# Create input data
X = np.random.randn(2, 10, 8)

# The second example is of length 6 
X[1,6,:] = 0
X_lengths = [10, 6]

cell = tf.nn.rnn_cell.LSTMCell(num_units=64, state_is_tuple=True)
cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=0.5)
cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell] * 4, state_is_tuple=True)

outputs, last_states = tf.nn.dynamic_rnn(
    cell=cell,
    dtype=tf.float64,
    sequence_length=X_lengths,
    inputs=X)

result = tf.contrib.learn.run_n(
    {"outputs": outputs, "last_states": last_states},
    n=1,
    feed_dict=None)


print(result[0]["outputs"].shape)
print(result[0]["outputs"])
assert result[0]["outputs"].shape == (2, 10, 64)

# Outputs for the second example past past length 6 should be 0
assert (result[0]["outputs"][1,7,:] == np.zeros(cell.output_size)).all()

print(result[0]["last_states"][0].h.shape)
print(result[0]["last_states"][0].h)

```

## Day 62: September 6, 2018

Finally able to figure out why the training of the RNN was taking so long. For one thing it was due to the length of the corpus. The size of the text file was ~2MB, but given that I am running it on an APU and not a dedicated GPU it takes a while. I tried to run this in the Udacity workspace with GPU mode enabled and it took me ~35 mins to go over 20 epochs. That shows you how compute intensive this is going to be. Now I am worried that the project will eat up most of the compute time I have.
f
One workaround for this that I did, just for the sake of trying to see how it would look like is by editing out the text file. I limited it to until chapter 20 and I also reduced the epochs to 10. Even with that my PC is still _struggling_ with the size. __I really need my AWS account.__

I am also reading about the contrib layer where RNN belongs to in TensorFlow. There are some good reading materials that explains the process but seeing as this is still a contrib layer, TF has not released yet a digestible guide. But this is far better off than doing it in Keras, where from what I have read the layers for RNN are not only contributed but already under maintenance for quite a while. But that shows you the power of community, there is someone out there willing to give time to create layers for future users and researchers. Although it should be noted that creating layers from Tensorflow is far better than just using and waiting for the Keras layer to come up. I have not yet come across information regarding Theano as Keras backend.

So I was able to do some digging. Cutting out the file to save time will severely limit the results that you will get out of the system.

## Day 63: September 7, 2018

Learning to use the contrib layers of TensorFlow for RNN today. I am checking out if Kaggle can do the RNNs for me. If it can then I can use the GPU to leverage my speed in training.

I think it is possible. I forked a Kernel about stock prices and RNN using LSTM in Kaggle. Quite interesting actually. I forked it to study and to learn more about stock predictions becuase its acutally part of a project that I am thinking about doing.

__tf.contrib.rnn class__

Here we will try to go over the documentation for the layers of RNN available in tensorflow. The most basic cell is `tf.contib.rnn.RNNCell`. Based on the documentation, this is the basis of all the succeeding cells used in RNN class: LSTM and GRU cells. Properties of this would be `output_size` for the output produced by the cell and `state_size` which is the size of the state used by the cell.

Following this [guide](https://www.tensorflow.org/versions/r1.0/api_guides/python/contrib.rnn#Core_RNN_Cells_for_use_with_TensorFlow_s_core_RNN_methods) we next have the Core RNN cells for TensorFlow's core RNN methods.

`tf.contrib.rnn.BasicRNNCell` which as the name implies is the most basic RNN cell. Not much else on the documentation sadly. As it is based on the rnn.RNNCell, it also has the same properties (output_size and state_size). Arguments include `batch_size` and `dtype`.

`tf.contrib.rnn.BasicLSTMCell` is the equivalent basic RNNCell. Also has no solid documenation behind it. Arguments include `num_units` which are the number of units in the LSTM cell (vertical stack?). `activation` is also  a possible argument which is set by default to tanh. Note that this can only be editted via the `__init__` method.

There are also wrappers in for RNN cells. To work you first need to have craeted beforehand the cell. What the wrapper will do, from what I understand, is that it will take the original cell and apply another argument to it (for example dropouts).

`tf.contrib.rnn.MultiRNNCell` which will create a sequential RNN cells of the same type as the simple RNN cell that was defined. Essentially it will provide us with the columns of our model (i.e. Left to right width) while the basic cells defined earlier will define the depth(i.e. top to bottom).

`tf.contrib.rnn.DropoutWrapper` which is used when you want the cell to have dropouts in an input and output. I think you can use this to create one cell and then from this one use MultiRNNCell method to create the model.

[Practical Guide for RNN in TensorFlow and Keras](https://paulx-cn.github.io/blog/4th_Blog/) is outlining what to do. Also this [post](http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/).

## Day 64: September 8, 2018

Shut down all other things and focus on the project. I have to finish the project over the weekend.

I was curious on the Word2Vec embeddings and found the image below which is from [part 1 of a tutorial](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/). So what is the logic behind it? How does it work? The most basic idea is that it will output the collection of the words that are __in context__ with the given word. That is, words that are mostly seen in front or behind the given input. For example we have the word ant: the expected outputs would be hill, colony, queen, red, black, worker. As we can see, this are words that we normally assume are related to ants: ant hill, red ant, black ant, ant colony. Do note that this is still dependent of the training corpus that the model used. The context learned by the model is dependent, as always, with the training data fed to it. If every occurrance of ant in the system is just 'red ant' and 'ant hill' then we will expect red and hill to have a higher probability but that the rest of the words will be according now to occurance rather than context. This is actually the logic behind data poisoning. By increasing the occurance of a word artificially we can, if we are trying to trick the model, just flood it with the same incorrect information and the weights will adjust accordingly.

![http://mccormickml.com/assets/word2vec/training_data.png](http://mccormickml.com/assets/word2vec/training_data.png)

Now in the image above, the model is a simple one where the window is just 2 steps forward and back. The input for the model is the word highlighted in blue. So for the first one the input is 'the' and the training samples it got were (the, quick) and (the, brown). With this, it will update the weights. Then it will go on to the second pass where the input is now on 'quick'. With this there is now one step backwards in the window. The pairs received are (quick, the), (quick, brown) and (quick, fox). This goes on and on until it reaches the end of the training corpus.

With the idea in mind, I think I now get the issue with why training the chanracter wise RNN in the `intro-to-rnn` topic takes so long. It actually has to come up with probabilities character wise instead of word wise. Sure there are lesser possible characters than words (this is due to the number of possible combinations and permutations of letters that make up a word) but there are more characters in a training file than words.

Now on to [part 2](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/) of the tutorial there is insights on how to optimize the word2vec model. There are three ways that were cited from the original source [paper](http://arxiv.org/pdf/1310.4546.pdf). One optimization put forward was the use of "paired words" as single words or phrases for the model to reduce complexity. For example the words 'new', 'york' and 'city'. It is quite obvious for humans to only come up with 'new york city' from this set of words. It is also highly likely that any mention of a set containing these words are going to be arranged as 'new york city' since the training data will have to have been written by humans (for now :imp:). How do tell the machine to learn it this way? The solution put forward was for the machine to treat pairs of words or group of words comming together often as a single word. Obviously it would be difficult for the machine to immediately learn from a 3-word set. So the obvious, and also scalable, solution is to find first a pair of words that can be combined as a single word. So in this example we will have 'new' and 'york' being combined first. In this case we will have a new word 'new_york'. Then as the model passes again, it will figure out that 'new_york' is closely related to 'city' so by then it will combine the two words together to form 'new_york_city'.

Another solution put forward in the paper is the use of subsampling frequent words to reduce the number of training examples. The idea behind it is quite similar to the dropout conecpt where we want to make the system more resilient by randomly turning of some of the nodes. In the case of subsampling, instead of randomly turning of the nodes we actually remove words based on their frequency. For example, we have the word 'the' in the example: `the quick brown fox jumps over the lazy dog`. As you notice, the word 'the' appears more frequtenly than the other words (granted it appeared twice lol). This in turn would have created duplicates in the training samples we have produced if we have a big window. By doing subsampling, we can tell the model to ignonre the word 'the' in some of its passes so that we do not form a bias and overfit on it as well as reduce the number of passes our model has to do.

The last solution put out in the paper was the use of 'negative sampling'. On this solution it addresses the requirement of updating all the weights for a given word. Using word2vec will give us a lot of parameters to train and update, this is due to the unique relationship between words. For example, (the, quick) and (the, brown) are both unique pairs and this is just for a simple single sentence. Imagine if your training corpus is the entire wikipedia or reddit comments section, then you have to deal with thousands of pairs. Since this is a  neural network, we have to also have hidden layers which constitues another set of weights. So by now we have a weight matrix with a size of a thousand pairs by the number of hidden layers. This would easily push our weight matrix to the millions level. The way negative sampling works is that instead of updating ALL of the weights as is common with neural networks, we simply select a number of words that we want to update for example 5 words. Only the weights of the paired words and negative words selected will have to update their weight matrix. The rest will simply be updated as zero and the implementation would be do nothing. This is similar to the way Trusk did it in the sentiment analysis at the earlier part of the degree on the introduction to NN.

## Day 65: September 9, 2018

So today, I am going over the project. Reading over the specifications and the assertions in the `unittest.py`.

So here is a sample exchange of script for Simpsons that is in the original text file:

```text
Barney_Gumble: Hey Homer, how's your neighbor's store doing?
Homer_Simpson: Lousy. He just sits there all day. He'd have a great job if he didn't own the place. (CHUCKLES)
Moe_Szyslak: (STRUGGLING WITH CORKSCREW) Crummy right-handed corkscrews! What does he sell?
Homer_Simpson: Uh, well actually, Moe...
HOMER_(CONT'D: I dunno.
```

I was able to do the project and here is a sample of the script that was generated:

```text
moe_szyslak:(nasty laugh) ah, ha ha, you got me, didn't ya?(handing homer a beer) all right, here you go--" red tick.
lenny_leonard: oh, i'm so sorry...
moe_szyslak:(generously) aw, homer. you know, i gotta check with lenny on that. uh, swishkabobs.
waylon_smithers:(knowing) yeah, right. it's too bad. could have been fun. could have made a little money.
moe_szyslak: thank you for saving my precious... gheet!(sighs)
moe_szyslak:(sympathetic) aw, that's a procedure. you're talkin' about deadly, life-threatening surgery, here.
homer_simpson:(getting nervous) why, homer, you're doin' great. you're way ahead in the polls... even got a lot of catching up to do.
homer_simpson:(noise of pain, then) can i just get a glass of water?
moe_szyslak: water.
carl_carlson: to smithers, the little cutie wants to do something cute...(to barflies) shut up, ya bums, have a right now.
lenny_leonard: that's why i'm not doing?
lenny_leonard: oh, sure thing, homer. i learned how much of my blood and sweat are in this drink?
barney_gumble: good for you, moe. only an idiot would give away a million.
moe_szyslak: oh, yeah. here ya go.
moe_szyslak:... then we light a match... and fwooof! we start a new life in hawaii.
moe_szyslak:(sobs) i'm happy on christmas eve. and for once it's not 'cause some drunk left a wallet in his other skirt, and he pays me with this!
carl_carlson: hey, i don't know, carl. he might be closer than you think.
homer_simpson: moe, what are you doing?
moe_szyslak:(sighs) what's the point?... same ol' stinkin' world... ah, your tavern... here comes the evening rush. clear out, fellas.
barflies: what a day. / let's get started. / some serious drinkin'.
homer_simpson:(chuckles)
homer_simpson: no, uh, hey, moe. you've got a job here for your sign.
barney_gumble:(gives a little wave) yoo hoo!
moe_szyslak: oh, god, no.
```

I have made a submission for this one. I am currently waiting for it to be reviewed but for the most part, I was able to meed the expectations of the rubrics. I also asked some questions regarding the placeholder name as I am still getting some issues with it. For now, I will continue learning.

__GANs: Generative Adversarial Networks__

From what I understand in the intro video GANs are generally like mirrors trying to create an oytput as close to the input as possible. This is a naive way of explaining GANs

## Day 66: September 10, 2018

Received the review. Its a pass. In terms of unit test here is a [post](https://docs.python-guide.org/writing/tests/) on python unit testing. The reviewer was kind enough __to provide a [link](https://datascience.stackexchange.com/questions/11402/preprocessing-text-before-use-rnn) to the Data Science Section of StackExchange__. There is this [link](https://stackoverflow.com/questions/36693740/whats-the-difference-between-tf-placeholder-and-tf-variable) about the difference in `tf.placeholder` and `tf.variable` which is a good supplementary read. He even went on to provide a suggestion on how embeddings could have been done better via `tf.contrib.layers.embed_sequence(input_data, vocab_size, embed_dim)` method. An [indepth explanation via article](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) is also given, which is the same as the ones in the lessons so I have been able to go over this one. There is [this guide](https://www.tensorflow.org/api_guides/python/nn#Recurrent_Neural_Networks) in TensorFlow regarding embeddings and building Recurrent Neural Networks. I got a commendation on my choice of keeping the hyperparameters to a power of two. This makes it easier for TensorFlow to compute them more efficiently, being able to provide background as to why those values were chosen was also appreciated. I think [this link](https://magenta.tensorflow.org/) is the best of all in terms of impact. Its the link to Project Magenta by Google/TensorFlow. They are currently using Music and Art using Machine Learning. The possibilites are amazing.

Now on to GANs. The lecturer for this course is the creator of the GAN model himself, Ian Goodfellow. So exciting to learn the basics from this guy. As far as I can tell he is good at making you understand what the concept is via his examples. In terms of the components of the GAN there are two main ones: the _discriminator_ and the _generator_. Their functions are pretty self explenatory. The logic of how they function is the basis of GAN. In simple terms the generator tries to come up with ways to pass the discriminator and the discriminator looks for ways to discern the outputs from the discriminator. The example given was about the discriminator being the cops and the generator being criminals who want to counterfiet money. The cops has to come up with ways to tell which ones are fake money and which ones are real. The criminals look for ways to fool the cops into thinking that their output is real.

The basis of the GAN concept was actaully based on Game Theory. The underlying assumption is that for all possibilities between a scenario, there would be a point where in all the distribution of possibilities would lead to a zero-sum game (?). The example here was the rock-paper-scissors game. Basically, the scenario is that two players will try to play the game: They correspond to our discriminator and generator. In this example let us consider we are the discriminator.

## Day 67: September 11, 2018

More on GANs. Since the connection is acting up again, I did some readings of papers on GAN. First one on the list if the seminal paper by Ian Goodfellow et.al. Mostly the background and the idea of the paper first.

## Day 68: September 12, 2018

Watching the lessons on GAN. Ian had some recommendations regarding how a Discriminator and Generator in GAN can be implemented. He also gave out some caution regarding the counter-intuitive ways to implement GAN that for some reason works well in implementation. Also started watching GAN implementation by Mat. In terms of the paper on GANs, I have not yet finished reading a 9 page paper. Really busy on work lately.

Any way here is an update regarding the readings on GAN. Building on the context earlier about the criminals and cops. The GAN is trained through the _adverserial_ portion of the network. Meaning that the descriminator will have to reduce errors by finding out the outputs of the generators from the actual outputs and the generator has to minimize loss by finding a way to fool the descriminator into thinking that its output is the real thing. This will go on, in theory, until the least ammount of loss and error is acheived.

In terms of the adverserial network, the idea is that it will be like both Descriminator (D) and Generator (G) playing a minmax game. Now obviously if there is no initial bias on the network, it will encounter an issue where in the Descriminator is going to get more and more into the negative side that it will not be able to find a way to minimize loss. Imagine the descriminator as the police wherein obviously upon learning of the counterfeits it won't immediately be able to tell. When it receives the report of a counterfeit, it would already be out so it would incur a loss. Then this would happen for some time until they are able to train on the previous data and know how to detect the counterfeit. Once the descriminator is able to correctly detect the fakes all the generator has to do is change one thing on the fakes and the police will have to figure out again what changed. This is one of the algorithms behind Adeverserial networks, we have to __alternate__ between _k_ steps in the descriminator and _one_ step for the generator. This is veiwed from one of the perspective, another way to look at it is that the descriminator ealry on will just descriminate __all__ the samples as fake with high accuracy since the generator will, for lack of better word, suck. Given that the generator can only update after _k_ steps and knowing that none of its previous attempts made it through the descriminator then it will lead to complications because it will not be able to learn anything. *Feels familiar*.

## Day 69: September 13, 2018

Now on to Intro to GANs notebook. The idea is to train a GAN on MNIST dataset. Basically, we want to generate hand written numbers after our trainings.

For more information on GAN applications we can view them [here](https://github.com/wiseodd/generative-models). Here is an example of a GAN application in action, this is called CycleGan where a video of a horse was transformed to make it appear like a zebra.

![CycleGAN - application](https://raw.githubusercontent.com/junyanz/CycleGAN/master/imgs/horse2zebra.gif)

For more readings on some GAN applications, we can use [this one](https://arxiv.org/abs/1611.07004) about Pix2Pix project. Or this [Medium post](https://medium.com/@jonathan_hui/gan-some-cool-applications-of-gans-4c9ecca35900) about applications of GAN. Here is a sample [paper](https://arxiv.org/pdf/1804.00064.pdf) on GAN for Dental Restorations.

Back to intro to GAN. First up we define our placeholders for input and z.

## Day 70: September 14, 2018

Working again on the intro to GAN lectures. While trying to figure out the `tf.variable_scope`I was directed to this [guide in TensorFlow](https://www.tensorflow.org/guide/variables#the_problem). What's great about this guide is that it also covers items that are essential for us, for example it covers High Level APIs like Keras, it has a section for Checkpoints and Estimators and also Low Level APIs like Tensors and Variables and Graphs. All the things that you might need to get a good grasp of Tensorflow can be found in the [guide section](https://www.tensorflow.org/guide/).

So back to the variables scope documentation. I was reading about it and from what I understand, it is simply wrapping the variable created to an implicit scope. Its quite similar to global and local variables I think. One way variable scope is needed is when we are doing multiple layers of NN work. In this case we may need to get multiple outputs of the same name. I am copying the example of a point where the variable will fail since tensorflow does not know what behaviour is expected. In the case below, we are trying to name a variable and an operation while calling an earlier variable. The first variable x is called as an input to the second layer while the output is still called x.

```python
input1 = tf.random_normal([1,10,10,32])
input2 = tf.random_normal([1,20,20,32])
x = conv_relu(input1, kernel_shape=[5, 5, 32, 32], bias_shape=[32])
x = conv_relu(x, kernel_shape=[5, 5, 32, 32], bias_shape = [32])  # This fails.
```
To solve this issue we can define the scope of our varialbes:

```python
def my_image_filter(input_images):
    with tf.variable_scope("conv1"): # Same names but on different scopes
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu(relu1, [5, 5, 32, 32], [32])
```
In here we can share variables by setting the argument `reuse = True`:
```python
with tf.variable_scope("model"): 
  output1 = my_image_filter(input1)
with tf.variable_scope("model", reuse=True):
  output2 = my_image_filter(input2)

# OR Alternativeley we acn use scope.reuse_variables()

with tf.variable_scope("model") as scope:
  output1 = my_image_filter(input1)
  scope.reuse_variables()
  output2 = my_image_filter(input2)
```

So in the case of the mini-project, our generator function will have a scope of `generator` and with `reuse` argument set to `False`.

## Day 71: September 15, 2018

I am going over the Deep GAN lectures for today. Goal is to reach at least 25% of the lecture for now. Then I will go back to the intro to gans notebook. I am really procrastinating right now. But I know I am close. I can feel it. From nothing to GANs in 70 days is quite great. I want to finish this now and move on to building my portfolio. There is a plan behind this one. I just have to recover the initiative.

For Deep Convolutional GANs, the end goal is to generate Street View House Numbers. This lesson will cover also semi-supervised learning with GANs. The idea is that we will try to fool our discriminator by creating an image that closely resembeles the real input. To do this we will need to do convolution. For our generator protion we will need to use the transpose mode to create the image. Recalling from CNNs, we have a wide image with depth of 3(RGB). For our generator architecture we are simply going to reverse the order of the layers, instead of going from a wide and shallow to small and deep we will go the other way. Sort of similar to the encoder decoder problem. Some key points to take in this is that we do not use any fully connected layers (since we are not after some features), we also activate via ReLU for all layers. We use transpose with strides to increase the size of our matrix and we will use batch normalization as well.

For the discriminator we are going to use something similar to the CNN because we are going to look for features in the images. Remember that the goal of a descriminator is to find the features of the images (both real and generated) so it must have a CNN in this case. We still have ReLU as our activation function for our layers and we will have Batch Normalization as well.

_Batch Normalization_ is a technique for improving performace and stability of neural networks. How it works is that we normalize the layer inputs so that all items will have a mean of zero and a variance of one. This is quite similar to using standardize in network inputs. The use of batch normalization is __necessary__ in ensuring that our DCGANs work. Recalling what happened in the CNNs dog breed classifier project we now that we can increase the speed of training for the network for a slight decrease in the accuracy. Since GANs will use both generation and discrimination function with CNN as the base, we expect that the computations would be intensive. In this case, the use and understanding of how to setup batch normalization is critical.

## Day 72: September 16, 2018

__"Rebellions are built on hope"__. This is one of the main things I wanted to do which led to me to the path of AI, [bots](https://chatbotslife.com/crypto-trading-bot-on-raspberry-pi-3-using-profittrailer-and-bittrex-41c2d63e3697). Actually, its the idea of automating the most boring of things so that you can free your time. Its tied in to the concept of _toil_ in devops, you should allocate a toil budget for your resource wherein they can and are expected to engage in toil. Tasks that consume toil time with less benefits should be candidate therefore for automation. Also, another article this time for [High Frequency Trading Bot](https://www.indiehackers.com/interview/building-a-3-500-mo-neural-net-for-trading-as-a-side-project-5dda352c13). I am intending to make this my Capstone project for the degree. Might be feasible and lucrative.

## Day 73: September 17, 2018

So no coding done yesterday. Today is also the same, minimal coding only. My focus for now is to go over every readings necessary. I have to read about Batch Normalization and its implications on GANs and on training as a whole. Then I will touch on the different bots out there, dabbling on what they are and how we can make them.

First of let's build upon the previous lessons on batch normalization. _Normalization_ should be familiar to us as it is already used in the input portion of our model. All our data, as much as possible, are normalized, this is basic normalization. The idea of _batch normalization_ is that instead of treating the entire model as a whole, we think of it modularly (? is that correct). What we mean by this is that we deconstruct the layers of the models and treat those as individual input components with input and output arguments. If we view the model this way, we can see now that there is an opportunity to deploy normalization in between the layers. Do note that the layers will go over by batch, this is the reason for the naming. This is the intuition behind the use of normalization, there are also mathematical ways on which the batch normalization approach is defended. For more readings, the book of Ian Goodfellow et.al about [Deep Learning](http://www.deeplearningbook.org/) and also this [paper](https://arxiv.org/pdf/1502.03167.pdf).

So we now move on to the benefits of batch normalization.

1. __Faster training speed__ - counter-intuitively, the training is going to be much more faster even if we added more hyperparameters and calculations during the forward and backward pass. The speed difference is actually because the model, even if it has to calculate more items, can make up for the time by the relative ease of the computations as a whole (i.e. the covergence of the actual model happens faster).
2. __Possiblity of higher learning rates__  - The idea behind gradient descent is that we need to have a small learning rate so that the nework will converge. As the networks get deeper, the gradient actually gets smaller (vanishes). With the application of batch normalization we can actually use a higher learning rate and still allow the network to converge.
3. __Weights are easier to initialize__ - Not really in a sense that we can just throw any initialization but batch normalization allows us more freedom in choosing what our initial weigths are going to be and still allow our network to converge.
4. __Allows for more activation functions__- This is in the context of giving us more activation functions to choose from. We all know that the problem with sigmoids and ReLUs and Leaky ReLUs is that they often die out too quickly.
5. __Simplifies creation of deeper networks__ - As a by product of the earlier items, batch normalization allows us to create deeper layers which is always good. Caveat: the idea is that layers that are 2 to 3 stacks are normally good but more than that and the returns almost do not justify the cost. So, deeper in this context would be in terms of the nodes inside the layers and not the layers themselves.
6. __Provides regularization__ - Bach normalization is known to add noise to the network. It has been known to work as well as a dropout (the randomness is equated in this case to noise). So we can, in general, consider batch normalization as a bit of extra regularization so we can remove some of the dropout we might add to the network.
7. __May provide better overall results__ - With the addition of the batch normalization in every interconnection of layers we actually add more parameters which should slow down the network. But in terms of the speed of training/convergence, it has been shown that this is simply not the case. Also, since we will be using normalization, we can build deeper models that is often a good thing. To put simply, Batch Normalization is to be treated as an optimization to help us train our network train faster.

>Personal Opinion - Think of batch normalization as a capacitor in your input/output layer. Instead of dropping off immediately, the capacitor will try to maintain the levels so that the circuit will keep on working. This is almost )but not really) similar to the function of batch normalization.

In terms of creating a pipeline for development, I think creating a class or at least defining functions will make it easier to make the code modular for further reuse. For example we can create functions for read_csv, for weight initialization and for hyperparameters. Most of this items will be repetitive and can be used across multiple projects with minimal code change. That is it for now.

## Day 74: September 18, 2018

Note to self: review how to actually use graphs in TensorFlow as it makes it easier to understand the training methods and the way we training works in terms of the changes in the loss and accuracy or any other metric that we might want.

Two ways to do batch normalization in Tensorflow. The first one is `tf.layers.batch_normalziation` and the second one is  via `tf.nn.batch_normalization`. Based on the modules they are taken, we can see that one is a higher-level function while the other is for more lower-level works. `tf.layers` is the one for higher-level implementations, it will usually work for most of our problems and use cases but it also pays to learn about the `tf.nn` method in case we want a more controlled use case.

This is the reference for [implementing and testing Batch Normalization in TensorFlow](https://r2rt.com/implementing-batch-normalization-in-tensorflow.html). Also, the Batch Normalization Lesson in the nano-degree does offer an explanation on how it works and its limitations.

With all of these, I can now consider the topic of Batch Normalization as completed. In case I have any other work to do or need some refreshers I think I have enough materials to scour over again.

Now I am going on a tangent and read about bots for now. Procrastination. Here are some of the resources that I found which I have not yet touched:

[Viber Build a Bot](https://github.com/Viber/build-a-bot-with-zero-coding)

[100 Best Github Chatbot](http://meta-guide.com/software-meta-guide/100-best-github-chat-bot)

[Viber Bot with Python](https://github.com/Viber/viber-bot-python)

[ChatterBot](https://github.com/gunthercox/ChatterBot)

This is a post on towardsdatascience for [Learning DS when you are broke](https://towardsdatascience.com/how-to-learn-data-science-if-youre-broke-7ecc408b53c7) which is simply a guide on how to tackle on DS. The next phase of my training for now would be to create 2 capstone project. One would be a Kaggle Competition and the other would be a Deep Learning Bot. Besides this one, I also need to tidy up my github and add my project codes for Udacity projects.

[Fast.ai season 1 episode 22: Dog Breed Classification](https://towardsdatascience.com/fast-ai-season-1-episode-2-2-dog-breed-classification-5555c0337d60)

[8 Machine Learning projects for beginners](https://elitedatascience.com/machine-learning-projects-for-beginners)

[ML-AI Case studies](https://towardsdatascience.com/september-edition-machine-learning-case-studies-a3a61dc94f23)

WildML's [2017 AI and Deep Learning year end review](http://www.wildml.com/2017/12/ai-and-deep-learning-in-2017-a-year-in-review/)

[PSEi data](https://www.johndeoresearch.com/data/)

[A post on PSEGet](http://pinoystocktrader.blogspot.com/2010/11/amibroker-charting-software-chart-data.html)

Okay, so I was reading [this post](https://www.indiehackers.com/interview/building-a-3-500-mo-neural-net-for-trading-as-a-side-project-5dda352c13) about a trading bot and there are some interesting insights I have learned.

## Day 75: September 19, 2018

Reading more on Batch Normalization and GANs. Job for today is to watch semi-supervised GANs. So I have just finished Ian Goodfellow et.al. paper on GAN, just figured out on the testing part of their paper that the intended enemy of the _G_ is the _D_. What hit me is that it now makes sense that most of the images are, at least for human eyes, are somehow weird or off.

They do have some resemblance but when you place an unknown person and claim that he looks like this guy then you are already biased I think, especially if we all know what the original looks like. Can we imagine, for discussion, when the time comes where we are able to interface with the machines directly? If we are able to be the descriminator. One problem with this is that, at least how I see it, our generator will only be as good as our discriminator and nothing more. That is how I see it, for new things to emerge I think it has been pointed out in history that limits are always going to be exceeded.

Here is the tutorial page for [Quantopian](https://www.quantopian.com/tutorials/) which is for stocks and finance analysis and data. Then we have this for [TensorFlow](https://www.tensorflow.org/tutorials/). Areas of interest: for now GANs, then the possibility of building it for mobile because why the hell not.

Watching semi-superviesde learning by Ian Goodfellow. So far we have only used GANs for image generation which is still a useful direction for GANs with multiple AI initiatives tackling on this. Another way we can use GANs is for semi-supervised learning. This is a more general use of GANs and an example use case of this would be to improve the performance of classifiers. It should be pointed out that not all AI initiatives require the Generation of images, a more general use case would be using classification than generation. So the focus now for classification is the _Discriminator_ instead of the generator. In the earlier model where we have generated faces, the output of the discriminator is fed to a sigmoid function where the images are classified as either real of fake. In semi-supervised learning we instead use softmax activation so that it can now classify among a variety of buckets via probabilities.

The example we would be using is a classifier for SVHN dataset. We will be classifying the numbers between 0-9 and another bucket for all the fake images. A normal classifier would only be able to correctly classify from _labeled images_. For semi-supervised GAN application we would be able to classify between _labeled images_ , _fake generated images_ and _unlabled images_. This is now more in line with the current industry usage for classification because labeled data is expensive. By leveraging the power of GANs on unlabled images we can now use the vast unlabled data taht is available in the internet. One additional technique that will be reuqired for this setup is for the use of _feature matching_. Here is my understading of feature matching: Our generator would tend to overfit our discriminator because it wants to always pass the test. The problem here is that most of the outputs of the genrator would then tend to look the same which would lead to the discriminator catching on. In feature matching, the idea is that the statistical disctirbution of a feature is considered as well  before the classification layer. Normally, this feature is one of the deep layer nodes right before the softmax activation so we now that the features here are already deep and rich in a sense (since it has already been extracted multiple times, remember that the deeper the layer the more specific the feature is). So the idea is that the real images will also have a probability distributions with respect to this feature. Feature matching will therefore try to reduce the Mean Squared error between the distribution of the real image feature and to the generted feature. In simple terms, its going to also match the distribution (ideally) of a certain feature when generating an image.

>Looking back to the RNN model, I think that this is one of the reasoning behind the `random_choice` method that was used during the pick word function. Instead of just picking the highest probability among the words, we are actually randomly choosing based on the probability of the words appearing. This way the output is more stable. WOW. Insight. :muscle: The problem before in the script generation was that after a while the same words or sequencees were being repeated over and over to till the required length. By actually using the probability of the words in choosing randomly we achieved a more stable result. Stable in the context that we were able to move away from the repetitivesequences or words. So, for future reference, if we are able to get the probabilities of a feature then we should try to at least choose randomly with respect to the inherent distibution of probabilities and not just default to the argmax function.

## Day 76: September 20, 2018

Started out the Project for today. Went on to code some parts of the project: placeholders for the images and the learnrates and also started on the discriminator functions. Currently reading on the [DCGAN paper](https://arxiv.org/pdf/1511.06434.pdf) for more insights.

## Day 77: September 21, 2018

Still on the DCGAN paper. Right now I am trying to figure out what Architecture I would want for this project. It looks like its going to be an hourglass shapeed architecture (obviously) with the Generator feeding to the Discriminator. Actually, it seems like it should be inverted hourglass(?) or diamond. The generator is increasing in size and the discriminator is decreasing in size.

[LINK FOR DCGANs](https://towardsdatascience.com/implementing-a-generative-adversarial-network-gan-dcgan-to-draw-human-faces-8291616904a)

## Day 78: September 22, 2018

Now, doing the Generator function. A few more and we can begin the training. I am facing some issues with GitHub again, I can't open most of my notebooks.

So far, i am done with my generator. Up next is the training optimizer function.

## Day 79: September 23, 2018

Objective for today is to submit one draft before 6PM. So for now we are at the model_loss function. it will get the loss for both discriminator and generator. This is one of the unique traits of GAN, since we have two systems working we need to have two losses to track as well. Then after we do the `model_loss` we proceed with defining our `optimizer` function for training. For this one, again, we are using __Adam__ as our optimizer.

## Day 80: September 24, 2018

Managed to submit the project file for initial checking. I am having a problem with the generator. It is not initializing well. For some reason there are ghosting in the first steps. So I sent the file for assistance from the reviewer on how to fix the issue.

The review was fast, although it was not that helpful in terms of insights. I am about to apply the changes but for now the connection is again bad. What else? I received an email again from EDS team about algorithmic trading. Its a sign. :joy: I am still working on this nano-degree and I intend to finalize my GitHub Repo for the whole course before moving forward.

For now I am going to apply the necessary changes in the code and I will make another training run for it. Hopefully that will fix my issues. The goal for now is to have the initialization of the images to be black which means that there are now biases or ghosting for the model.

## Day 81: September 25, 2018

Currently playing around with my hyperparameters now. I reset my alpha values for the generator to be the same as the discriminator at 0.2. I have also adjust my layers in the generator with 512 256 256 instead of the 512 256 128. I figured I might as well make use of my GPU time :imp:. For now I still have 40+ hours of GPU enabled time in the course and I assume the quadcopter project will not take that long. One entire training pass for the DCGAN face generation project takes approximately 25 mins to 30 mins balancing out the production of an MVP should be of importance. Viewing the [guide](https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/contrib/eager/python/examples/generative_examples/dcgan.ipynb) I am amazed by the idea of using a GIF to portray the changes while it is training, like a flipbook of some sorts. I still have to play around with it for a bit but I like the idea of being able to animate what is actually happening along the way.

So I just received my second review. It is now more clear what the problem was and how to fix it. By now the main reason I cannot get better results is that I have some lousy hyperparameters. I am going over the hyperparameters selection again and I will be doing some initial runs in the office later. I was able to make one run earlier but my results were not that great and my batchsize was smaller so I generated more images than I wanted to. I have to fix this. One of the more pressing issue I am facing is the use of Dropout layers as suggested by my reviewer. Basing on [this post](https://towardsdatascience.com/dont-use-dropout-in-convolutional-networks-81486c823c16) it says that we do not use dropout in between Convolutional layers, and I am guessing that since Conv_transpose is just the opposite then we also do not want a dropout there. Normally, dropouts are used between Fully Connected layers so in this case the Dense layer. This also has to be fixed.

In terms of discriminator bias (pun intended :joy:) we have in place a smoothing factor. We use to balance out (ironically by providing bias) our discriminator loss function. This is done to help the generator overcome the initial strenght of the discriminator. Without smoothing, there is a possibility that the generator will collapse during training and it will simply stop trying to beat the discriminator.

I have also added Xavier intialization to the kernel initializer. From the reviewer's tip it says that it helps in the initialization of the weights. Reading further, I found out that Xavier is used to keep the intial weights to within a Gaussion (? if I remember correctly) distribution. This in turm would help the model during the training as it will avoid the gradient from blowing up uncontrollably or from going down to where it vanishes and the model gets stuck.

Main issue right now is actually in the hyperparameters. Beta values, alpha values, learning rate and batch sizes all would play a part in the training of the model. In a numbers perspective, I think the celebA training actually performs good with the Generator loss being lower than the Discriminator loss which would indicate that the generator is actually able to pull the discriminator. The only problem is that the faces are somewhat uncomprehensible when you look at them. This is a problem if we submit because strictly speaking we have a rubric that requires our generator to have some comprehensible faces generated. I have no other concerns with the MNIST data set. It does not really have a problem in a way because numbers have less features than faces so we need to fix first the celebA generation and the MNIST generation would (if I am correct) be fixed as well.

## Day 82: September 26, 2018

So I found this [post](https://towardsdatascience.com/implementing-a-generative-adversarial-network-gan-dcgan-to-draw-human-faces-8291616904a) in towardsdatescience with the implementation of a DCGAN. It had within it the use of Dropouts in between the Conv2D and Conv2DTranspose. It is just outputting black boxes for the MNIST data. This really is a bit frustrating, the concept of GAN is easy to grasp but damn the implementation is really difficult. The balancing of hyperparameters to come up with a result is hard and often counter-intuitive.

So I finally passed the Face Generation project. :muscle: It does have some more improvements to be made but overall the reviews came great. All units passed. Actually I want to download my reviews if possibe so that I can refer to them again as notes. There are lots of instances where I have to learn more about hyperparamenters and how they interact with the model. Building GAN was fun (frustrating but fun).

## Day 83: September 27, 2018

Here is the review from the project page, I am saving it for future reference:


Meets Specifications

Well Done!!! You have met all the specifications, but don't stop here, keep experimenting. Experimenting is the only way you understand DL.

Go ahead and explore the wonderful world of GANs. Below are a few links for starters...
1) In order to gain more intuition about GANs in general, I would suggest you take a look at this link.

2) Also, if you want to gain intuition about the convolution and transpose convolution arithmetic, I would suggest referring to this [paper](https://arxiv.org/abs/1603.07285).

3) For more advanced techniques on training GANs, you can refer to this paper.

4) One of the biggest problem GAN researchers face (you yourself must have experienced) with standard loss function is, the quality of generated images does not correlate with loss of either G or D. Since both the network are competing against each other, the losses fluctuate a lot. This problem was solved in early 2017 with introduction of Wasserstein GANs. With WGAN, the loss function directly correlates with how good your model is, and tracking decrease in loss a good idea. Do read it up.

5) Finally, have a look at this amazing library by Google for training and evaluating Generative Adversarial Networks.

6) Here are some other important resources for GAN:
http://www.araya.org/archives/1183 for GAN stability.
https://github.com/yihui-he/GAN-MNIST, https://github.com/carpedm20/DCGAN-tensorflow for DCGAN.
https://medium.com/@ageitgey/abusing-generative-adversarial-networks-to-make-8-bit-pixel-art-e45d9b96cee7

7) Below are few GAN videos:
https://www.youtube.com/watch?v=dqwx-F7Eits&list=PLkDaE6sCZn6FcbHlDzbVzf3TVgxzxK7lr&index=3
https://www.youtube.com/watch?v=RvgYvHyT15E
https://www.youtube.com/watch?v=HN9NRhm9waY
https://www.youtube.com/watch?v=yz6dNf7X7SA
https://www.youtube.com/watch?v=MgdAe-T8obE

8) Take a look at this Progressive Growing of GANs for Improved Quality, Stability, and Variation, which creates HD quality photos similar to the below image.

All the best for your future and Happy Learning!!!

 Required Files and Tests

The project submission contains the project notebook, called “dlnd_face_generation.ipynb”.

The iPython notebook and helper files are included.

All the unit tests in project have passed.

Great work! All the unit tests are passed without any errors. But you need to keep in mind that, unit tests cannot catch every issue in the code. So, your code could have bugs even though all the unit tests pass.
Build the Neural Network

The function model_inputs is implemented correctly.

Correct, you have defined the placeholder tensors, which are the building block in computation graph of any neural net in tensorflow.

The function discriminator is implemented correctly.

Correct implementation of Discriminator, good work!

The function generator is implemented correctly.

Good Job implementing the generator!

Experiment with more conv2d_transpose layers in generator block so that there're enough parameters in the network to learn the concepts of the input images. DCGAN models produce better results when generator is bigger than discriminator. Suggestion: 1024->512->256->128->out_channel_dim (Use stride as 1 to increase the number of layers without changing the size of the output image).

The function model_loss is implemented correctly.

Correct!
Good job utilizing label smoothing for discriminator loss, it prevents discriminator from being too strong and to generalize in a better way. Refer https://arxiv.org/abs/1606.03498

The function model_opt is implemented correctly.

Correct.
To avoid internal covariant shift during training, you use batch norm. But in tensorflow when is_train is true and you have used batch norm, mean and variance needs to be updated before optimization. So, you add control dependency on the update ops before optimizing the network. More Info here http://ruishu.io/2016/12/27/batchnorm/

Neural Network Training

The function train is implemented correctly.

* It should build the model using model_inputs, model_loss, and model_opt.
* It should show output of the generator using the show_generator_output function

Great work combining all the functions together and making it a DCGAN.

Good job scaling the input images to the same scale as the generated ones using batch_images = batch_images*2.

Tip: Execute the optimization for generator twice. This ensures that the discriminator loss does not go to 0 and impede learning.

Extra:
1) Talk on “How to train a GAN” by one of the author of original DCGAN paper here..

2) Here is a post on Gan hacks, https://github.com/soumith/ganhacks

3) Plot discriminator and generator loss for better understanding. You can utilize the below code snippet to plot the loss graph to get a better understanding.

```python
d,_ = sess.run(…)
g,_ = sess.run(…)
d_loss_vec.append(d)
g_loss_vec.append(g)
```

At the end, you can include the below code to plot the final array:

```python
Discriminator_loss, = plt.plot(d_loss_vec, color='b', label='Discriminator loss')
Genereator_loss, = plt.plot(g_loss_vec, color='r', label='Generator loss')
plt.legend(handles=[ Discriminator_loss, Genereator_loss])
```

You'll be getting a graph similar to the below image,
![image1](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/84774/1537935964/graph.png)

The hyperparameters chosen are correct and your model generates realistic images. Good Job!

You can further improve the quality of the generated image by experimenting with the parameters and the tips I provided in generator. Below are a few extra tips on choosing the hyperparameters for starters...
Tips:

1) Try using different values of learning rate between 0.0002 and 0.0008, this DCGAN architectural structure remains stable within that range.

2) Experiment with different values of beta1 between 0.2 and 0.5 and compare your results. Here's a good post explaining the importance of beta values and which value might be empirically better.

3) An important point to note is, batch size and learning rate are linked. If the batch size is too small then the gradients will become more unstable and would need to reduce the learning rate and vice versa. Start point for experimenting on batch size would be somewhere between 16 to 32.

Extra: You can also go through Population based training of neural networks, it is a new method for training neural networks which allows an experimenter to quickly choose the best set of hyperparameters and model for the task.

Below is an output that I got by modifying based on the tips. Experiment, that's the only way you learn Deep Learning.

![image2](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/84774/1537935991/download_6_.png)

Your model generates good face images and hyperparameters are correct. You can still improve your model to generate realistic faces by following the same tips I provided for you in the above MNIST section.

Tip: If you want to generate varied face shapes, experiment with the value of z_dim (probably in the range 128 - 256).

Below is one of the generated images that I got by making slight changes to your model.

![image3](https://udacity-reviews-uploads.s3.us-west-2.amazonaws.com/_attachments/84774/1537936048/download_5_.png)

So, there we go. A very long review which is good beause I now have more materials to look forward to. I am interested to learn also about the plotting/graphing part of TensorFlow because I am a strong believer in visuals and learn better when I actually see the graph than just going over the losses and infer from those numbers.

> Dropout layers is a regularization technique. We use it to prevent overfitting. With a smaller data set (~1000 or less), we would want to reduce overfitting if we want to create a more universal model. This is the importance of dropouts and it is not the same as batch normalization either.

A tip on _feature matching_ is that for convolution classification we do not use batch normalization, instead we use another form of normalization or none at all for code simplicity. Actally implementing batch normalization when we do feature matching kind of defeats the purpose of the feature matching. This is because the batch normalization will subtract off the mean of of every feature value and add an offset parameter that it learns as a bias. When we subtract from the mean then essentialy our feature matching will no longer work as there is no difference in averages. A reasearcher from OpenAI had a workaround for this via weight normalization which would be a good read.

So I am watching now Lecture 13 - Generative Models from the Standford course. Generative is apparently a distinct subset of networks where it is expected to generate an output (like seriously) based on the given inputs. For example, RNN was actually generative, in a sense that it can produce a label or a similar text based on the initialized input.

## Day 84: September 28, 2018

Finished the GAN semi-supervised learning videos. Summary to come up next.

So a recap on to the GAN semi-supervised learning videos. The idea behind it is that as semi-supervised GANs can leverage the lack of data by generating new ones. This in turn would be used to add to a limited number of data which our classifier would then use to build its model. So from the original small dataset we train our GAN to generate sample data that is close to the original one. Then from this augmented dataset we then train our classifier to figure our how to correctly classify the data. The unsupervised part comes due to the fact that the fake data generated would be unlabled. This is concept that I had in mind and how I understood the semi-supervised model. The good thing with this is that you can use it to come up with a sturdier model with less data. As we already know, having data is already expensive. Having labeled data is going to cost so much more. This semi-supervised learning model using GAN is a great tool in overcomming some of the difficulties that we may encounter due to the lack of useful data. This is my summary of what I learned from the lecture. I am now moving on to the supervised learning module. The final module. I have come a long way from where I was before this foundations degree. But looking forward, I have a long way more to go and that is exciting.

## Day 85: September 29, 2018

We are on standby duties today. So more of videos and readings and lectures for today. I see a lot of topics on the Reinforcement learning module. Most of them are 30 minutes to an hour long. I am guessing that they are more of explenatory videos and lessons on how to apply Reinforcement learning and what we can do with them.

Now on reinforcement learning. Basically, the modeling of learning through interaction: We prove that nobody was born a master but we can learn through interaction and learning what each and every action corresponds to  (positivie or negative). This is basically the concept of reinforcement learning. Interact with the world, did it help? If it does then we get a reward and we note what we did. Interact again, did it help? If it does not then we lose something but we also know what not to do. This goes on and on until we learn enough things to be able to do more advanced task. And so on until we can actually model how to do complex tasks.

First we need to define the words that we will likely encounter throughout this course. __Agent__ is the doer that we want to train, it will be the one interacting with the environment. To give it more context, we would want to think of the agent for this case to be a _puppy_. Then we want to train our puppy, we give the command "sit". Obviously, the puppy would have no idea what the command initially means but it knows it can do a lot of basic things like "sit", "bark", "roll over", etc. Since he still has no idea we would assume that he will pick one action on equal probability at random. For simplicity, let us say our puppy did sit, then we reinforce that response by giving it a treat (or if we are heartless we electrocute it for every wrong response). So by now the puppy will have a small bias to "sit" more when it hears the command "sit". Repeatedly do this loop of action reward and we can form a habit which is what we want to achieve for RL. For simplicity again we would assume that our agent/puppy is only interested to maximize the rewards. So every time he responds correctly to the command he gets a reward then instinctively his concept of action-reward will get reinforced.

The idea above is far simplistic and I will be able to learn more about how complex learning actually is. For example, the puppy should be able to learn as well pre-requisite actions that are not commanded but is necessary to get the reward. One way this is explained is that suppose the puppy already knows that when he performs "sit" he will get a reward, the dog could try to just sit there and expect to get a treat but actually it still has to learn that it will only need to sit when the pre-requisite command "sit" has been given. This is just one of the things that our agent would need to learn, and this will only be learned through interaction to the world.

Moving on, we are introduced to the tool that we will mostly be using to test our models for RL: __OpenAI Gym__ :joy:. This [blog post](https://blog.openai.com/openai-gym-beta/) from the openai website is the announcement and simple introduction to OpenAI. Simply put, its a toolkit where we can test out our models/algorithms in trying to solve Reinforcement Learning (RL) algorithms for various environments. Now the environments of OpenAI is the place where our agent will interact and will learn what to do and what the objective is going to be.

Here is the [docs](https://gym.openai.com/docs/) for OpenAI gym. We can read through it to get the installation and setups done.

## Day 86: September 30, 2018

Continuation of the RL series. I am on standby today and the message notifications are off so I am going to be up until possibly 5-6AM. Now we go deeper into the cycle of interaction between the _agent_ and the _environment_. This is very similar to _FEECONT_ in university, feedback and controls class. In this case we are creating a general model which can be the basis of how we approach a RL problem in the future. So the idea is simple, we have the agent and the environment. The agent observes the __State (S)__ of the environment and takes __Action (A)__. Once the action has been completed, based on the reaction of the environment the agent will then get a __Reward (R)__ which can be postiive +1 which is a reward or negative -1 which is a penalty. Then it will go on and on in doing this with the intention of getting the most number of rewards. In the case of the puppy agent we had earlier, he will learn to not just map out the command to the correct action but also to learn when the action is needed to be done and do the action only when the command is given (do nothing is an action).

Now we go on to learn about the _sparse reward_ problem in RL. This happens when the agent only gets rewarded at the end of each episode. For example the agent plays chess. With episodic model in place, we only get rewarded after the game is over (i.e. checkmate). Now should already be able to see what the problem is, your agent has to go through the entire length of the game before the reward based on the outcome is claimed. It can do bad things while in the game and still win and we will be reinforcing the bad things it does because it does not know it is bad. In the context of winning, its all part of the plan to get the reward. Also, our agent could have been doing brilliant moves along the way and still lost and it disregards those moves for future reference because it has led to a loss. This would eventually lead to problems early on and the agent can simply collpase and stop training unless we change our algorithm. This is a case of perspective: Do we look at the task itself as one big outcome or do we look at the task as the accumulation of multiple steps that could then be optimized to minimize the total costs to reach the objective?

Tasks in RL can be episoding or continuing. _Continuing_ tasks are tasks that will go on forever without end. Episoding task on the other hand are tasks that have a well defined starting and ending point. In this context, we define an _episode_ as the complete of sequence of interaction from start to finish. With this defenition, we can then get the other assumption that episodic tasks come to an end whenever the agent reaches a terminal state.

The reward hypothesis is the next topic. The idea of giving reward to agents is borrowed from Behavioral science. The terms reinforcement and reinforcment learning also comes from behavioral science. The idea behind giving reward is that we want to promote actions taken by our agent that is contributing to the achievement of a goal so that it is more likely to do these actions in the future. But what constitutes a reward? For our example with the dog, it is simply giving a reward after the correct action is done based on the queue. This is a straight forward reward approach. But, complex problems like teaching a robot to walk properly can have different interpretations of when a reward is due. Do we give a reward if the robot is able to move from point A to point B? Do we encourage the user with rewards when it is able to move without falling. A lot of interpretation can be made. But the gist of the reward hypothesis is:

> All goals can be framed as the maximization of __expected__ cumulative reward.

We then move on to the DeepMind [paper](https://arxiv.org/pdf/1707.02286.pdf) about using reinforcement learning to train locomotive behaviours for a humanoid (i.e. Walk and move past obstacles). What we need to learn here is how to factor in the objective to be maximized in the way we give rewards to our agent. First we have to define the setup of the sytem. The agent is controlling a humanoid, the objetive is that it has to make it walk. The feedback is taken from various sensors in the joints of the humanoid as well as a contact point which when triggered would indicate that the humanoid has fallen and will terminate the episode. The whole episode is then defined as the entire length from which the humanoid is standing until the humanoid falls and triggers the termination. We can see here that for the humanoid to get the most ammount of reward, it has to stay upright and keep walking for as long as possible. So how did they define the reward for the agent? What they did was first they decided to keep the reward equation consistent and simple across all terrains that the humanoid has to go over. The main component of the reward is a funtion of the velocity along the x-axis of the humanoid so that the humanoid would be enticed to move forward, then there is a small term that adds penalties on excessive torques. (context: the humanoid is moving along the x-axis for forward and backwards movement, y-axis would then mean that the humanoid is moving up or down (i.e. climbing a step or jumping), and z-axis is when the robohumanoidt has to move laterally when it has to avoid obstacles or walls for example). The reward equation was then modified to include penalties for deviating laterally (i.e. go straight as much as possible), another penalty for the deviations vertically (i.e. do not jump uneccessarily). So the final equation for the humanoid reward (for time t) is a function of a reward proportional to the velocity (set to a maximum), a penalty for the lateral deviation, a penalty for the vertical deviation, a penalty for excessive torques on the joints and finally a fixed value small reward which is to encourage the humanoid to at least stand properly. If we read the reward funtion correctly, the idea is now clearer on what the expected behavior is going to be. It cannot be seen here as I have not given the proportions in this post but the idea is the same. For one, the agent has to keep the humanoid to at least stand up to get the constant reward in the reward function. Second, the humanoid will have to learn how to move the humanoid forward to get the reward for the velocity forward. Now that the robot is able to get the idea of what to maximize, it then has to minimize the penalties it incurs. First, it has to apply the correct force to its joints so that the torque penalty is reduced. Then it has to learn when to jump over obstacles (like hurdles and falls or steps) and when to avoid obstacles and go around it (i.e. when walls are in place). It looks quite simple when it is now given as a model, but reaching the proper tuning is going to be the difficult part.

## Day 87: October 1, 2018

So just a continuation of RL for today. Been watching the DeepMind video by ColdFusion in YouTube. They are on to some great things. I am just happy to be pivoting towards that. For now we have the continuation of the problem of Reinforcement learning. We have covered now the concept of discounting and _discount factor_. The _discount factor_ basically allows the agent to plan its actions some steps in advance but in proportion to a multiplier. The discount factor is denoted by gamma (small letter y, I need some Latex) and is always in the range between 0 and 1. The discount factor allows the agent to model some steps in advance but understandably through a limited ammount of time. This is important feature to have because our agent needs some way to know that there is a future and that there is potential reward for future actions so that it does not unnecessarily terminate itself. This works due to the logic behind the reward hypothesis in which the agent has to maximize the __cummulative__ reward. By letting the agent know that there is a possibility for future reward, albeit at a discounted rate, it will be able to plan more and try to reach longer. The gist of the discount factor is that the larger the discount factor is, the farther the agent is looking ahead. Generally, discount factors are set closer to 1 than 0 because choosing a small discount factor will cause a fault in the agent where it will try to terminate itself because it cannot see the potential future reward afforded by the higher value discount factor.

Then we touched on the conecpt of Markov Decision Process (MDP). This is where we are able to model the different states, the actions that can lead to these states, the probability that a certain action is taken, and the reward for the provided action once completed. In MDP we have a __state space (S)__ which is the set of all (nonterminal) states. In cases of episodic task, we use S+ to refer to the set of all states which includes terminal states because it is already episodic. Then we have the __action space (A)__ which is the set of all possible actions in the model. One-step dynamics determines how the environment decides the state and reward at every time step. One-step dynamics is the condition probabilities for each possible next state (s'), reward (r), current state(s), action (a) which is p(s',r | s,a).

So now I am finished with the statement of the RL problem. I am on to the RL solution in Udacity. I have 10 modules to do and so far I have done 2 of them, at the end there is a project as well. I might go beyond the 100 Days for this but that should not matter for now. The target is to crunch in the remaining time. There is still time, 12 days is long enough even if I do this one module a day I still have, probably, enough time to take on the Quadcopter project.

Now, the evening session of Day 87. Wow. :joy: First let me just internalize it. 87 days. 87 days of doing something towards Deep Learning and AI and Machine Learning. 87 days since I decided to pledge an hour everyday to learn, code and do something related to the AI-ML-DL field. Day 4 was when the official start of the Deep Learning Foundations Nano-degree began as part of my __#100DaysOfMLCode__. Now I am on the last stretch. I am not yet there but I can see the end and, in the context of discount factor, the possible future I can have in the field. To think that from some negative thought would come and nudge me in this direction. It really is great to continue looking forward and not dwelling on the things that are already out of your control. Acting out on the only available choice left for you. If ever someone would read this, thank you for reaching this far. You can do it as well. I don't know why you would even bother looking at someone else's logs but if you are: You are awesome! :smiling_imp: :muscle:

So, to continue on the momentum I am now reading up on the Introduction To Reinforcement Learning from the book [_Reinforcement Learning: An Introduction 2nd Ed._](http://go.udacity.com/rl-textbook) by Richard S. Sutton and Andrew G. Barto. I am not trying to read all of it, its 548 pages. I am just going to read the introduction and the basis of RL.

> Reinforcement learning is learning what to do—how to map situations to actions—so as to maximize a numerical reward signal. The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them. In the most interesting and challenging cases, actions may affect not only the immediate reward but also the next situation and, through that, all subsequent rewards. These two characteristics—trial-and-error search and delayed reward—are the two most important distinguishing features of reinforcement learning.

The excerpt above is taken directly from the book. I want to just point out that Reinforcement learning does not have any initialized bias or pre-coded reactions that is hidden behind it. The agent starts with a clean slate and only through constant trial-and-error interaction as well as the delayed reward condition will it be able to get an idea of what it has to do.

## Day 88: October 2, 2018

Okay, So now we are at Module 3. We are introduced to _policies_, _state-value functions_, _Bellman equations_. In terms of policy we have two types: _Deterministic_ and _Stochastic_. For deterministic, there is exactly one mapping to the action based on the state and its probability is 1. Meaning, the agent already has a script to follow for a given state. Stochastic on the other hand allows us to provide more actions to the current state. Since we now have multiple actions for a state, the next action is then determined by probability and not certainty (p=1, i.e. deterministic). For one of the Bellman equations (there are 4 in total, chapter 3 of the book) it simply states that the value of any state is the immediate reward + the value of the state that follows. This equation is recursive and allows us to determing the value of the current state in the environment. This is under the assumption that our discount factor is 1, therefore no discounting. But if we rewrite this to accont for the discount factor, then it becomes: the value of any state is equal to the sum of the immediate reward  and the discounted value of the state that follows. This Bellman Equation becomes the Bellman Expectation Equation, which is the one we will generally be using moving forward. This is because in complex worlds, the immediate reward and next state cannot be know with certainty.

Defenition The policy $\pi' \geq \pi iff v\sub{\pi'} (s) \geq v\sub{pi} (s) for all s \in S$. There is a possibility that there would be two policies which cannot be compared (i.e. the same policies). Also, with this in mind, it is guaranteed that an optimal policy exists although it may not be unique (i.e. Multiple paths that can lead to the same cummulative reward). The _optimal policy_ is denoted by $v *$ (v-star) which indicates that it is the optimal policy or an optimal policy.

We then move on to _action-value policy_. The state-value policy earlier gives out the value of the state $s$ under a policy $\pi$. Action-value policy is giving the value of taking action $a$ in state $s$ under a policy $\pi$. In state-value policy (svp) we denote it with a single value on the tile, this is because the action is predetermined. For action-value policy (avp), we now plot all the values for the possible actions from a given state that leads to the terminal state. Again, as usual, terminal states will have a value of 0 because there is no action done. The notation for action-value funtion is q and the optimal action-value is denoted by q*.

How do we boil this down in relation to the agent? Basically, the agent will interact with the environment, it will try out different routes and plot the rewards and state-value as well as action-value. When it is done with the interaction, it can then proceed to estimate the optimal value function. Then from that optimal value function, it can estimate the best policy to choose. So, once the agent decides the optimal action-value function q*, it can then proceed to obtain the optimal policy $\pi *$ by getting the argmax of the action-value for a given state. The question that remains now is how does the agent interact with the environment and decide how to estimate the action-value for the state.

Lightly writing about Thinking, Fast and Slow: Chapter 22. This is a very interesting chapter as it speaks about intuition, skill and how it is acquired. If you read it, you will see the connection between how an agent learns viewed through a lens of psychology and behavioural sciences.

## Day 89: October 3, 2018

Grinding for more videos to watch in the Udactiy course. As this is more of a lecture intensive module, I am guessing that I might have to exceed the estimated time.

Now we are on DynamicProgramming and we will be using OpenAI Gym: FrozenLakeEnv. The story of the FrozenLakeEnv is pasted below and copied from the [Github Repo](https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py):

```

Winter is here. You and your friends were tossing around a frisbee at the park
when you made a wild throw that left the frisbee out in the middle of the lake.
The water is mostly frozen, but there are a few holes where the ice has melted.
If you step into one of those holes, you'll fall into the freezing water.
At this time, there's an international frisbee shortage, so it's absolutely imperative that
you navigate across the lake and retrieve the disc.
However, the ice is slippery, so you won't always move in the direction you intend.
The surface is described using a grid like the following
    SFFF
    FHFH
    FFFH
    HFFG
S : starting point, safe
F : frozen surface, safe
H : hole, fall to your doom
G : goal, where the frisbee is located
The episode ends when you reach the goal or fall in a hole.
You receive a reward of 1 if you reach the goal, and zero otherwise.
```

The idea behind it is quite simple, our objective now is that we need to write our code in a way where we would be able to make our agent train on the environment. But that is to come, for now we will focus more on how to arrive there. Now let us try to apply an iterative approach on how to find the Bellman Expectation Equation. So first a review of Bellman Expectation Equation is that it is simply the probabilty of the action mutliplied by the sum of the reward for that action plus the Expectation value at the succeeding state. So for multiple possible actions from the current state, it is simply the summation of the Bellman Expectation Equations to all the available actions/state pairs. So with this in mind, we can now define all the equations for all the possible states in the environment. This would then lead us to a system of linear equation for all the possible states. For simple systems like a 2x2 grid environment, this can be bearable to manually compute as there would only be 4 equations and 4 unknowns. But the use of the _iterative method_ is for systems that are much more complex especially for the much larger MDPs. So how does the _iterative method_ work? Basically we need to start with an initialization. In this case, we are going to assume that all the values are initially zero. From this we will then solve for the values of the states, do note that intuitively we would start at states near the goal where there is no need for the terminal state and work our way back. We would then check if there is a big difference in the changes of the states. We then go over all the possible states checking if there is still a big change in the values before moving on to the next step. This is where the iterative part comes in. The idea behind this is that as the iteration gets deeper, the change in the value of the state gets smaller and would always converge (this is because there will always be a solution and optimal path). Once the change between the states get small enough then we can already assume that it is the _True value_ for the state.

![Iterative Approach Psuedocode](https://d17h27t6h515a5.cloudfront.net/topher/2017/September/59cc184c_policy-eval/policy-eval.png)

The policy evaluation as laid out in the iterative approach is guaranteed to converge to the state-value function coresponding to a policy $\pi$, as long as the state-value for that policy is finite for each state in the state space. In essence, as long as there is no loops in the system or as long as there is a defined path to the goal or terminal state then there would always be a policy that is optimal. For a finite Markov Decision Process (MDP), the convergence would be guaranteed as long as eitehr conditions below are met:

* Discount factor is less than 1 (therefore there is a need for the reward to decay) or;
* if the agent starts in any state in the State space, it is guaranteed to eventually reach a terminal state if it follows policy $\pi$

Some additional pointers on the convergence conditions. The idea behind the first line where the discount factor is always less than 1 is to discourage the system from getting into a loop that will not terminate. If we simply add up all the states while in the loop then the agent would think that by going around, it will gain more rewards but it is no closer to the terminal state since it is just doing the same thing over and over again. Also, one way to look at it would be that since the reward is not discounted, the agent will simply try to pass by all the available states before going to the terminal state. By discounting the rewards, the agent will be forced to shorten the states that it has to pass through so that the reward at the end of the state will not get severely discounted. The second line would mean that there is some state in the State space where the agent will start that will not have a path towards the terminal state when it follows a policy. This is for those unique cases where there is no continuity in pathing between the start state and the terminal state or when the policy from that state will force the agent to go into a non-terminating loop. In both these conditions, if they are not met then we can say that the iterative policy evaluation will not converge because the state-value function may not be will defined.

The example from the lecture regarding this condition is stated below:

Consider the following:

* Two states s1 and s2, where s2 is the terminal state.
* One action a (This is also a Markov Reward Process since there is only one action)
* The probability of going to s1 with a reward of 1 from s1 doing action a is 1.

The conditions above is talking about two states s1 and s2 with s2 being the terminal state but the only action for s1 is a which directly leads back to s1. In essence, s2 is isolated. So in this case, the episode is non-terminating given the policy. If the discounting factor is 1 then the value of state s1 given the policy is just going to be 1+1+1+... which is never going to converge, this will diverge to infinity. So in this case, the policy is the one that is not allowing the value-state to converge. As stated in the module, one optional way to learn about convergence is the review of geometric series and negative binomial distribution.

```python
while True:
    delta = 0
    for s in range(env.nS):
        Vs = 0 # Inintialize the Vs to be zeros
        for action, action_prob in enumerate(policy[s]):
            for prob, next_state,reward,done in env.P[s][action]:
                Vs+= action_prob*prob*(reward+gamma*(V[next_state]))# Update the Vs with the equation
                # action_prob is the probability that it will take that action according to the policy
                # prob is the probability that it will take that action out of all possible policy (meaning, no policy yet all fair)
        delta = max(delta, np.abs(V[s]-Vs))
        V[s] = Vs
    if theta < delta: # Check if there is still a need to update
            break
return V
```

## Day 90: October 4, 2018

Now we move on to _Estimation of Action Values_. This one is used for estimating V of the state-value function with the one-step dynamics of MDP and will return an estimate Q (action-value) for the policy. The pseudo-code for this is seen below. The input would be the State-value funtion and the output would be Action-value which is almost similar to the State-value but with the added multiplier of the probability for that action. This would be used to determine the values of the action-value for the environment.

![Estimate Action Values Pseudocode](https://d17h27t6h515a5.cloudfront.net/topher/2017/September/59cc021b_est-action/est-action.png)

```python
def q_from_v(env, V, s, gamma=1):
    q = np.zeros(env.nA)
    for a in range(env.nA): # Note that we are not going over the entire state for this one.
    # If we were going to do it for the entire state-space then we have to add another for loop taking s in range of env.nS
        for prob, next_state,reward,done in env.P[s][a]:
            q[a] +=prob*(reward + gamma*V[next_state]) # This is the updating of the action-value for a given action at a state
    return q
```

Then we have the _policy improvment_ section of the code. In this one, we are going to tell the agent how it would be choosing which action is going to be on the policy. In simplest terms, the idea is that the greatest value for the action-value would be the action that the agent would apply while building the policy. So in this case, we are sort of doing a shortest path/greatest action-value accumulation for the policy. This is will work because the action-values for each action is actually taking into account the reward and the state-value for the next-step so the calculation of the best action-value by the greatest value is valid. One caveat for this policy implemetation code is that it migh encounter a state where the action value are both equal (i.e. more than one path available). In these situations, the idea would be to arbitrarilly choose one and proceed with the policy evaluation. We would still be able to return back to the branching point later on while we check again for possible improvements of policy and we can then update the policy if necessary to get to the optimal one.

![Policy Improvement Pseudocode](https://d17h27t6h515a5.cloudfront.net/topher/2017/September/59cc057d_improve/improve.png)

For the policy improvement section, what we would need as input would be MDP and the value function of V and the output would be the value of the policy that is optimal (by that iteration).

```python
# Implementation of Policy improvement

#TODO: add implementation of Policy Improvement in here

def policy_improvement(env, V, gamma=1):
    policy = np.zeros([env.nS, env.nA]) / env.nA
    for s in range(env.nS):
        q = q_from_v(env, V, s)
        policy[s][np.argmax(q)] = 1 # Basically, just find the greatest action-value for the state and set the prob to 1
        # For Stochastic random equiprob
        #best = np.argwhee(q==np.max(q).flatten())
        #policy[s] = np.sum([np.eye(env.nA)[i] for i in best], axis=0)/len(best) # This is already stochastic not yet discussed
    return policy
```

Then we proceed with _policy iteration_.

For now, since I am unable to connect to the udacity workspace I am going to continue reading on the book on RL. One great point discussed in the book is on how machine learning fits in the supervised and unsupervised paradigm that exists on ML.

Another point that was breifly tackled but is propped up to be an ongoing problem in RL is the __exploitation-vs-exploration__ problem. Its just quite fascinating that I resonate so well with regards to this problem. The idea below is lifted from the book and I think there is some relatable points here that applies to life as well.

> One of the challenges that arise in reinforcement learning, and not in other kinds of learning, is the trade-off between exploration and exploitation. To obtain a lot of reward, a reinforcement learning agent must prefer actions that it has tried in the past and found to be efective in producing reward. But to discover such actions, it has to try actions that it has not selected before. The agent has to exploit what it has already experienced in order to obtain reward, but it also has to explore in order to make better action selections in the future. The dilemma is that neither exploration nor exploitation can be pursued exclusively without failing at the task. The agent must try a variety of actions and progressively favor those that appear to be best.

You have to know what you are good at but in order to know what you are really good at means you have to try things and accept that you are not good at it. Its simple really, when you think about it but it is hard to find the balance of staying put and moving forward to the unknown.

## Day 91: October 5, 2018

Good news for me [PyTorch Scholarship from Facebook is open for application](https://admissions.udacity.com/apply/intro-dl-pytorch-scholarship). Sent my scholarship bid and now I am just waiting if I do get accepted or not. I want to take on the PyTorch course to gain more insights on what the difference is. But for now we return to the lecture.

The topic for today is _Policy Iteration_ which in its simplest sense is basically the combination of the _policy eveluation_ and _policy improvement_. It begins with a policy chosen from an equiprobable random policy. Now, with the policy choses, it can start with the policy eveluation step with the initalized policy. Then after the policy has been evaluated, it will go to the policy iteration process where it tries to see if there is a better policy to be tested and this would iterate through the possible combinations until it is able to find the optimal policy for the problem. Provided that there is a finite number of iterations then this method should be able to converge on the optimal policy.

![Policy Iteration Pseudocode](https://d17h27t6h515a5.cloudfront.net/topher/2017/September/59cd57e2_iteration/iteration.png)

```python
import copy # This will allow us to create a shallow copy during the iteration process.

def policy_iteration(env, gamma=1, theta=1e-8):
    policy = np.ones([env.nS, env.nA]) / env.nA
    ## TODO: complete the function
    while True: # repeat iteration until optimal policy is acheived
        V = policy_evaluation(env, policy, gamma, theta)
        new_policy = policy_improvement(env, V, gamma)
        if (policy == new_policy).all():
            break # end when the new_policy is equal to zero (optimal)
        policy = copy.copy(new_policy) # Store the best policy for now and move on to the net iteration
    return policy, V
```

Then we move on to the _Truncated Policy Iteration_ portion. What is it really? First of all we have to know that we can apply truncation in policy evaluation and policy iteration portion. First is that we apply truncation on the policy evaluation portion and then we apply this truncated policy evaluation to the policy iteration to acheive a truncated policy iteration.. Recall that the vanilla version requres that we acheive a difference between the state-value functions to be zero or at least a small number theta for us to determine that we have arrived at an optimal position. Truncated policy iteration is a version of the policy iteration where instead of trying to acheive a value very close to the optimal, we instead would like to ask the iteration to stop after some determine steps. We do not truncate the process, we truncate the iterations. Why would it work? It would work under the assumption that we are converging when we are evaluating our values during each iteration step. So after a few steps in the iteration process the values should already be in the correct ballpark of the values. With this information, we can then assume that even the rough estimate is already useful enough to compute the values and get a policy that is optimal. Think of it as early stopping in a Neural Network, during the latter part of the training of NN the changes would be small that it would seem that they are not moving but the weights are already near the values we want them to be and they are already useful with some degree of accuracy. The truncated policy iteration is useful in providing a fast way to arrive at a policy for the agent and reducing the computations needed to meet the required theta value or the near optimal policy.

![Truncated Policy Evaluation Pseudocode](https://d17h27t6h515a5.cloudfront.net/topher/2017/September/59cda462_truncated-eval/truncated-eval.png)

So, as we can see in the pseudocode for the policy evaluation, we are simply using a counter to record the number of sweeps we have done and we are no longer using the theta value as our stopping criterion.

![Truncated Policy Iteration Pseudocode](https://d17h27t6h515a5.cloudfront.net/topher/2017/September/59cda5ad_truncated-iter/truncated-iter.png)

In truncated policy iteration, instead of calling the policy evaluation that uses theta we instead call on the truncated policy evaluation function. Do note that the theta value is actually used to gauage if there is convergence on the values of the state. So for the stopping criterion (optimal policy has been found) of the truncated policy evaluation, it has to compare the difference between the updates in state-value with respect to the theta. Now, there is a catch here. We should be carefull of the relationship between our theta and our number of sweeps. While it is true that the estimation is still going to arrive at the optimal policy, if we set our theta to a very small value, our loop could go on forever. Do note that the changes between every sucessive iteration is going to be larger in this case as it is on the earlier portion of the evaluation. This is similar to the early stages of training of a neural network, we should expect large swings of delta on the early stages so we should adopt a theta value that will be in line with these larger changes.

Then for the final part of the module, we are going over _Value Iteration_. This is the most condensed version of evaluation possible. What it basically does is that instead of evaluating the MDPs until the delta arrives at a small value (vanilla), or evaluating at fixed sweeps (truncated), what we do here is that we simply do __one__ sweep of evaluation of the state-vale and proceed directly to the policy evaluation. So in simplest terms, we are simultaneously performing policy evaluation and policy improvement. This would make it an estimate but it is fast.

![Value Iteration Pseudocode](https://d17h27t6h515a5.cloudfront.net/topher/2017/September/59cdf750_value-iteration/value-iteration.png)

And with this, I am done with the dynamic programming module. I am now moving on to Monte Carlo problems and the mini-project is a Blackjack Agent. So excited. For additional readings on the topic of dynamic programming, I need to do some readings on Chapter 4 of the book. Also, I have to pull my frozenlake AI and run it later.

## Day 92: October 6, 2018

Will take a break for this weekend. I am on to Monte Carlo problem for AI. Excited but I have other commitments. I would probably be watching some lectures and then update my notes at a later time.

Some readings about AI in stocks analysis [on a medium post](https://medium.com/@TalPerry/deep-learning-the-stock-market-df853d139e02). Here is [another one](https://medium.com/@jaungiers/how-to-develop-an-ai-trading-strategy-system-26ff8b1dcc35) concerning the development process.

I was able to watch some topics while on the bus earlier. Now we are on the Monte Carlo Methods for AI.

## Day 93: October 7, 2018

Recall first that we are working on an the interaction between the agent and its environment. The agent is in a _state_ and will do an _action_ which according to the rules of the environment is going to be given an appropriate _reward_ signal. The agent's goal is to _maximize the cummulative reward_. With this, the agent evaluates the _policy_ on what to do when interacting with the environment. So what is different in monte carlo methods? Monte Carlo Methods (MCM), unlike Dynamic Programming, does not require the model of the entire environment. It is the first learning method for _estimating_ value functions and finding optimal policies. MCM do not assume complete knowledge of the environment. What MCM requires is experience -- sample sequences of staes,actions, and rewards from _actual_ or _simulated_ interaction with an environment. So now, we have a method where we do not have to do prior calculations or know the layout of the environment. It can, through interaction, learn from the environment the consequences of its actions. While it is true that a model is required, the model need only generate sample transitions and not the entire list of probabilities for all possible transitions that was required in DP.

MCM is a way of solving the RL problem (interact with the environment and maximize the cummulative reward) based on averaging sample returns. One characteristic of MCM is that it considers the entire episode as the experience and increment in an episode-by-episode basis and not "real-time" or step-by-step. "Monte Carlo" is a broadly used term for an estimation method which operates with an involvement of _significant random component_.

So first, we have to learn how MCM solves for the state-values, suppose we want to compute for the state-value given a policy given a set of episodes (experience) obtained when we follow the policy and it passes state s. Each time we pass by the state s (think of it as the tile version) when we follow the policy is considered a _visit_. In the course of an episode, the state s may be passed several times. To better understand it, I'll put it in the context of cards for blackjack. State s can be the card "ace" and everytime we visit s would therefore mean that everytime we draw an ace. Now, back to the original discussion. We can compute the value of the state s based on the reward after following the policy but we have two options to go here. We can do a _first-visit MC method_ where the estimate of state-value for s as the average of returns following the first visits to S. Note that we used visits here instead of visit because remember that we are considering all the sample episodes and not just one episode. The second method for estimating would be _every-visit MC method_ where all the occurances of state s in an episode is considered and averaged.

![First-Visit MC method - State Value Pseudocode](https://d17h27t6h515a5.cloudfront.net/topher/2017/October/59dfe1e7_mc-pred-state/mc-pred-state.png)

Above is the pseudocode for the first-visit MC method. Do note that both first-visit and every-visit is guaranteed to converge. This roughly follows the __Law of Large Numbers__ concept.

```python
# TODO: Create a policy for the agent.
def generate_episode_from_limit(bj_env):
    # In this policy, the policy is that it will always HIT for values less than or equat to 18
    # It will always stick when the value is already above 18.
    # Do note that the dealer's policy is that it will stick at anything greater than or equal to 17.
    # Thus, we create a policy to beat that. Otherwise, we will just be breaking even by virtue of LLN.
    episode = []
    state = bj_env.reset()
    while True:
        action = 0 if state[0] > 18 else 1
        next_state, reward, done, info = bj_env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode
```

```python
# TODO: implement the value evaluation using MC method

from collections import defaultdict
import numpy as np
import sys

def mc_prediction_v(env, num_episodes, generate_episode, gamma=1.0):
    # initialize empty dictionary of lists
    returns = defaultdict(list)
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        # First, we get our experience by generating episodes
        episode = generate_episode(env) # Note that generate_episode is based on the policy we created
        # We prepre the SAR for that episode
        state,action,reward = zip(*episode) # unpack
        # Apply discounting
        discounted = np.array([gamma**i for i in range(len(reward)+1)]) # We are creating an array of the discount factor
        # Actual calculation of returns
        for i, state in enumerate(state):
            returns[state].append(sum(reward[i:]*discounted[:-(1+i)]))
    V = {k:np.mean(v) for k,v in returns.items()} # create a dict
    return V
```

Then after the state-value has been calculated, we can proceed with the action-value calculation. One notable corollary for the use of action-value computation for MCM is that we will not use if for estimating in deterministic policies. This is one of the things required to ensure that there would be convergence for the estimation. Do note that it is not really exclusively for deterministic policies but it is intended for zero-probability state-action pairs.

![First-Visit MC method - Action Value Pseudocode](https://d17h27t6h515a5.cloudfront.net/topher/2017/October/59dfe1f8_mc-pred-action/mc-pred-action.png)

Before we implement the action-value estimation using MCM, we first have to fix the deterministic policy we created earlier.

```python
def generate_episode_from_limit_stochastic(bj_env):
    # To make it work with MCM action-value estimation, we give it a stochastic policy.
    # In this example, we are using 80% probability that it will STICK to a sum of 18 and above.
    # While the cards are 18 below, it has an 80% chance to HIT.
    episode = []
    state = bj_env.reset()
    while True:
        probs = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]
        action = np.random.choice(np.arange(2), p=probs)
        next_state, reward, done, info = bj_env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode
```

Now that we have created a stochastic policy, we can now proceed with creating our action-value estimation.

```python
def mc_prediction_q(env, num_episodes, generate_episode, gamma=1.0):
    # initialize empty dictionaries of arrays
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        # The portion below is the same as before
        episode = generate_episode(env)
        state,action,reward = zip(*episode)
        discounted = np.array([gamma**i for i in range(len(reward)+1)])
        for i, state in enumerate(state):
            # This is where the similarities end. Below is the code for going over and estimating the action-value for the policy
            returns_sum[state][action[i]]+=(sum(reward[i:]*discounted[:-(1+i)]))
            N[state][action[i]] +=1
            Q[state][action[i]] = returns_sum[state][action[i]]/N[state][action[i]]
    return Q
```

We then move on to Monte Carlo Controls. The problem is, How does the agent learn the optimal policy by just interacting through the environment?

## Day 94: October 8, 2018

Was reading the book today. Not much else done today.

We now move on to how we can create a better method of updating the policy for MCM. Recall that earlier examples like the one we did for blackjack required that a sample of 500000 be created before the policy can be evaluated and improved. One solution put forward for this is to use the same concept as value iteration in DP. This would mean that we create moving averages of our value and update the policy every iteration. This would call for an incremental mean that we shall code below. How would this work? Say we have successive state-action-reward chains. We approximate the value of the state-value or action-value by doing incremental mean to the values as they appear in the sequence.

![Incremental mean Pseudocode](https://d17h27t6h515a5.cloudfront.net/topher/2017/October/59d6690f_incremental/incremental.png)

```python
# TODO: Incremental mean function
import numpy as np
def running_mean(x):
    mu = 0
    mean_values = []
    for k in np.arange(0, len(x)):
        # TODO: fill in the update step
        mu = mu + (x[k]-mu)/(k+1)  # Note that we are at index-zero. This is why we devide by k+1 as we would have to divide by zero if we did not use this.
        mean_values.append(mu)
    return mean_values
```

Monte Carlo Control Policy Evaluation. The incremental mean we have above is useful only for one pair. We have to make changes to it so that it can accomodate multiple state-action pairs. Now we move on to discuss _Greedy policy_. Recall first that to evaluate a policy of an environment that we do not know any idea about, we have to assume an equal probability for each action. The example put forward for this is in choosing between door A and B. Say we are trying to evaluate and improve our policy, since we have no idea of what the value for each door we simply assume that they are both zero initially. Since they are equal in value, we simply tossed a coin to decide which door we open. So for the first trial, we open Door B and we had no reward so the value are both still zero. Now we toss the coin again and we opened Door A this time. Then we had a reward of 1. Now the value of Door A is 1 and the value of Door B is 0. If we follow _greedy policy_ we are litterally going to choose Door A simply because it has a higher value that Door B. Now we opened Door A again and we got a reward of 3. Now the value of Door A is 2 while Door B is still at 0. So we choose again obviously, since we are greedy we opened Door A again and we got a reward of 1 again. This would go on and on and we will keep getting 1 or 3 everytime we opened Door A. If you are a simpleton then I guess you can call it a day and simply keep opening Door A. The problem with Greedy Policy is that it will tend to skew the values more to the one that has provided the initial value change. Now, there would never be a chance where we figure out that Door B has an equal distribution of 0 and 100 as rewards compared to Door A which has an equal distribution of 1 or 3 as rewards. It just so happened that the first impression we got from Door B was a reward of 0 and we immediately got a reward for Door A with reward 1. In hindsight and in this context, we would have chosen to open Door B forever because we already know what the distributiona and value of the rewards are. For the agent, which does not know and has to interact with the environment, this is not as straightforward as it seems. Remember that our agent does not have an idea of the distribution of values and rewards for each door and must rely on its interaction. Now we would want to solve this problem of the Greedy Policy being so rigid that it does not do anything else except open the proven action. We would want to introduce stochastics in the Greedy Policy so that even if it is still Greedy at heart, it will have a probabilty that it would pick the "un-greedy" action every now and then. To create this new version, we introduce a new value which is epsilon. Epsilon is a small number that we introduce to the probabilty of doing an action based on Greedy logic. Epsilon is a value between 0 and 1 and it indicates how often our agent would choose to explore and do the wrong thing. So now, with the introduction of epsilon, we are still in a majority going to follow greedy policy but with a chance of exploring the other option instead just so that we can check if there is somehting in the other option. This new version of greedy policy which has an epsilon value is called _epsilon-greedy policy_. The psuedocode for the epsilon-greedy policy is seen below.

![Epsilon-Greedy Policy Pseudocode](https://d17h27t6h515a5.cloudfront.net/topher/2017/October/59d53a9a_screen-shot-2017-10-04-at-2.46.11-pm/screen-shot-2017-10-04-at-2.46.11-pm.png)

Some things to look out for in epsilon values. One is that the value of epsilon can be between 0 and 1 _inclusive_. While having an epsilon value of 0 means that the greedy choice is always picked, it does not mean that having an epsilon value of 1 will make the agent always pick the non-greedy choice. This is better explained when you simply look at the pseudocode. Do note that it is still epsilon devided by absolute value of the action-value so it is not ALWAYS going to pick the non-greedy choice. In reality, having an epsilon value of 1 is simply going to make the chances equal for the greedy and non-greedy choice since the term left would be epsilon divided by the absolute value of the action-value. In short, as long as the epsilon value is greater than 0 would mean that there is going to be a chance that the non-greedy choice is taken (maximum of 50% probability).

I just remembered something with regards to Greedy Policy. During the RNN project of script generation, there was a piece of code there that almost acts similar to the greedy policy. I think that was on the next word function during generation where the RNN would choose the next word based on the probability. My first attempt there was that I just chose the argmax of the probability so that the highest probability won out. As per the reviewer's advise, I changed it to randomchoice based on probability. So the new model would be choosing between all the candidate next words but it is still based on their probability so the highest probability word would still have a higher chance of getting picked but the other words also has a chance. What happened to the RNN generated text was that it became more stable in the output. When I was using the argmax method, I noticed that the conversation actually became circular in nature and often would lead to a loop. With the addition of randomchoice, which we can think of in this context as epsilon-greedy, the results became more stable. _Just want to point out some of the idiosyncracy and counter-intuitive points I learned so far._

![Exploration Vs. Exploitation](https://d17h27t6h515a5.cloudfront.net/topher/2017/October/59d55ce3_exploration-vs.-exploitation/exploration-vs.-exploitation.png)
[source](http://slides.com/ericmoura/deck-2/embed)

Exploration-Exploitation Dilemma is a unique state on which our agent can encounter while solving the environments. Remember that our agent in this case does not have any pre-conceived idea of what the environment is and what the rewards are going to be. What it does know is what the objective is, which in this case is to win complete the task with maximum reward. Through interaction, the agent will become aware of how the environment works and bases will base the next action to take in the future based on the current experiences gained when interacting with the environment. Intuitively, we would devise a strategy(policy) for the agent to always select the action that it beleives (based on past experiences) will maximize the return. With this, the agent would most likely follow a policy that is greedy with respect to its action-value function estimate. While a policy following the greedy policy might converge to a policy, it can be a sub-optimal policy. This is beause the agent's knowledge is still limited and can be flawed. There is a chance, and a high chance, that the actions _estimated_ to be non-greedy (less value) is actually better than the chosen action based on the greedy policy. With this in mind, we should take note that our RL agent cannot act greedily at every time step. This would mean that it is always __exploiting__ its knowledge. In order to discover the optimal policy, it has to continue to refine the estimated return for all state-action pairs. This would mean that it has to be exploring the range of posiblities of values by visiting every state-action pair. The agent should always act _somewhat greedily_, towards its goal of maximizing return _as quickly as possible._ I think the quote _"Do you want to keep playing or do you want to win?"_ is appropriate here. This dilemma is the basis for the creation of the epsilon-greedy policy where instead of just choosing the action with the best value (based on limited experience), the agent will try to do actions that it thinks is sub-optimal for the chance of finding out if there is an update to that action.

How do we choose the value of $\epsilon$? There are two ways to look at this: Theoretical and Actual. First, we shall establish first that while exploration and exploitation is a dilemma for our agent, we should help it along by agreeing that at the early stages the agent should favor exploration and only after some time will it start to favor exploitation given that it has a good estimate of the action-value function for each state. Therefore, we shall aim to give our agent a policy that will favor first exploration and slowly lean more towards exploitation after some time. If we recall, setting $\epsilon = 1$ will make the actions equiprobable (50-50 chance of choosing between explore and exploit). Then we learned that $\epsilon = 0$ will make our agent _always_ choose the greedy algorithm (therefore exploit). With the first idea that we should slowly transition from explore to exploit and with the second idea of the values of $\epsilon$ and its realtionship with the behaviour of the system, we can therefore come up with the following values for $\epsilon$: Its value should start from 0 and slowly over time transition to a value of 1.

We then try to discuss about __Greedy in the Limit with Infinite Exploration (GLIE)__. These are actually conditions that will ensure that our MC control will converge to the optimal policy. The following are the two conditions defined in the GLIE to ensure convergence to the optimal policy ($\pi$$\scriptscriptstyle*$):

* every state-action pair $s, a$( for all $s \in S$ and a $\in A(s)$) is visitied __ininitely__ many times, and
* the policy converges to a policy that is greedy with respect to the action-value function estimate $Q$.

The condiions of GLIE there to ensure that the agent continues to explore for all time steps and that the agent will gradually transition to a more exploitive behaviours (but with the probability of exploring) at the later stages.

One way to satisfy the conditions would be to modify the value of $\epsilon$. To be specific, we have values of $\epsilon\scriptstyle{i}$ which corresponds to the $i$-th time step. To ensure that the GLIE is still met, we make sure that:

* $\epsilon\scriptstyle{i}$ $\gt 0$ for all time steps $i$, and
* $\epsilon\scriptstyle{i}$ decays to zero in the limit as the time step $i$ approaches infinity. This means that: $\lim\nolimits_{x\rightarrow\infin}\epsilon\scriptstyle{i}=0$.

In practice, however, we do not actually try to set and reach the value of 0 for our $\epsilon$. Even though the conveergence will not be guaranteed by mathematics, we can still get better results by either: using a fixed $\epsilon$ value or lettin $\epsilon\scriptstyle{i}$ decay to a small positive value, like 0.1. Why do we do this in practice? Whie the mathematical proof of convergence when we choose to let our $\epsilon$ value decay to 0 is guaranteed, what it does not tell us is _when_ the convergence would happen. Obviously, while it will happen, we do not have time to wait for it to converge. Convergence could happen in the first 1000 steps or in the 1 millionth step. The GLIE only assures us that it will converge. If we follow it and let our $\epsilon$ value equal to 0 then our agent will have a harder time correcting itself once it chooses the greedy choice even if the convergence has not yet been reached. This is the reason why we almost always never let our $\epsilon$ value be equal to 0. For more information regarding this, udacity gave us [this paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) to read.

Now we go on to implement our GLIE MC control for our blackjack agent. Do note that the one below is for first-visit MCM. We can change this to every-visti MCM and the results should still be the same.

![GLIE MC Control Pseudocode](https://d17h27t6h515a5.cloudfront.net/topher/2017/October/59dfe20e_mc-control-glie/mc-control-glie.png)

```python
# Additonal functions for reuse
# THis is taken directly from the solution manual.
def generate_episode_from_Q(env, Q, epsilon, nA):
    """ generates an episode from following the epsilon-greedy policy """
    # Taken from: generate_episode_from_limit_stochastic(bj_env) fnction
    # We need to create a blank policy instead of the earlier policy.

    episode = []
    state = env.reset()
    while True:
        action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA)) \
                                    if state in Q else env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode

def get_probs(Q_s, epsilon, nA):
    """ obtains the action probabilities corresponding to epsilon-greedy policy """
    policy_s = np.ones(nA) * epsilon / nA
    best_a = np.argmax(Q_s)
    policy_s[best_a] = 1 - epsilon + (epsilon / nA)
    return policy_s

def update_Q_GLIE(env, episode, Q, N, gamma):
    """ updates the action-value function estimate using the most recent episode """
    states, actions, rewards = zip(*episode)
    # prepare for discounting
    discounts = np.array([gamma**i for i in range(len(rewards)+1)])
    for i, state in enumerate(states):
        old_Q = Q[state][actions[i]] 
        old_N = N[state][actions[i]]
        Q[state][actions[i]] = old_Q + (sum(rewards[i:]*discounts[:-(1+i)]) - old_Q)/(old_N+1)
        N[state][actions[i]] += 1
    return Q, N
```

```python
# TODO: implement our GLIE MC Control for our agent
def mc_control_GLIE(env, num_episodes, gamma=1.0):
    nA = env.action_space.n
    # initialize empty dictionaries of arrays
    Q = defaultdict(lambda: np.zeros(nA))
    N = defaultdict(lambda: np.zeros(nA))
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        ## TODO: complete the function
        # set our epsilon value
        epsilon = 1.0/((i_episode/8000)+1)
        # NOTE: we can change the divisor from 8000 to a new value so that we can edit out
        # how much our epsiolon value updates.
        # Generate the episodes
        episode = generate_episode_from_Q(env, Q, epsilon, nA)
        # now we update our action-value function after the episode
        Q,N = update_Q_GLIE(env, episode, Q, N, gamma)
    policy = dict((k,np.argmax(v)) for k,v in Q.items())
    return policy, Q
```
One last lesson for the module and MCM will be done. We move now to constant-$\alpha$.

## Day 95: October 9, 2018

So why is there a need for constant-$\alpha$? Recall earlier that we tried implementing an incremental mean where we averaged the values calculated during our evaluation-iteration sequence. There is one minor flaw in using the incremental mean we have been using. Using it will mean that the early values will tend to have more weight on the mean than the later values. By recalling that the new values are the result of policy improvement, it should be clear that we would want the later values to have more weight in the average. Incremental mean will not allow us to do this because the later values would simply be diluted in the mean calculation. This is where the constant-$\alpha$ function steps in. Instead of calculating the mean for _all_ the values as they increment, constant-$\alpha$ will simply calculate the mean for a small window size taken from the recent calculations. Think of it as a _moving average_ where the latest values have more weight than the old ones. This helps us in solving the bias incremental mean would tend to have when it solves for the mean. Instead of taking to account all the values which would include the ones that are potentially flawed during the early stages of the policy, we simply take into account the latest values for computation so that the improved values will have more weight and give better inputs to the policy improvement.

![Constant-alpha evaluation Pseudocode](https://d17h27t6h515a5.cloudfront.net/topher/2017/October/59dff13c_screen-shot-2017-10-12-at-5.47.45-pm/screen-shot-2017-10-12-at-5.47.45-pm.png)

Looking at the psuedocode above, we can see that $\alpha$ is actually a coefficient or proportionality similar to discount rate. The idea behind $\alpha$ is that it will dictate the recency of the values to be considered.  $\alpha$ values would range from 0 to 1 and below is the implications for the values of  $\alpha$.

*  $\alpha = 0$ would mean that the action-value function is never going to get updated. (The agent will never update.)
*  $\alpha = 1$ would mean that the new value is going to be the new value. (The agent will always update.)

Let us look back on why the following implications exists. First would be when  $\alpha = 0$ the agent will never update. This would be because the difference between the estimate and the last value will never be added to the current value so the update funtion will not exists. For the second one, we simply have to distribute the signs and we can see that $Q(S\scriptstyle_t$, $A\scriptstyle_t)$ will simply cancel out leaving only $G\scriptstyle_t$ which is the current estimated value. So, smaller values of $\alpha$ will encourage the agent to consider a bigger window while a big value for $\alpha$ will make the user consider more recent values. $Q(S\scriptstyle_t$, $A\scriptstyle_t)$ $\leftarrow (1-\alpha)$ $Q(S\scriptstyle_t$, $A\scriptstyle_t)$ $+$ $\alpha G \scriptstyle_t$. The preceeding equation is simply a rewritten version of the original. It just shows how much the agent will trust the most recent value $G\scriptstyle_t$ as opposed to the estimate $Q(S\scriptstyle_t$, $A\scriptstyle_t)$.

There is an __important note__ that was given in the udacity lecture. There is no correct value of $\alpha$. Its setting would be largely dependent on the environment it is going to be used in so the trial-and-error method would still be the way to go. We have to take note that a high $\alpha$ will make the convergence to an optimal policy difficult because the values would be fluctuating due to the constant updates. Setting a low $\alpha$ on the otherhand will slow the convergence considerably.

![Constant-alpha GLIE MC Control Pseudocode](https://d17h27t6h515a5.cloudfront.net/topher/2017/October/59dfe21e_mc-control-constant-a/mc-control-constant-a.png)

The psuedocode above is going to give us the idea of how a constant-$\alpha$ evaluation would work in the MCM GLIE setting. Do note the that the psuedocode before this one is simply to show the evaluation of the $\alpha$. This current one is used to actually implement constant-$\alpha$ for MCM.

```python
# NOTE: additional function used to update the Q when using constant alpha
def update_Q_alpha(env, episode, Q, alpha, gamma):
    """ updates the action-value function estimate using the most recent episode """
    states, actions, rewards = zip(*episode)
    # prepare for discounting
    discounts = np.array([gamma**i for i in range(len(rewards)+1)])
    for i, state in enumerate(states):
        old_Q = Q[state][actions[i]] 
        Q[state][actions[i]] = old_Q + alpha*(sum(rewards[i:]*discounts[:-(1+i)]) - old_Q)
    return Q
```

```python
# TODO: Implement a constant-alpha GLIE MC control for our Blackjack agent
def mc_control_alpha(env, num_episodes, alpha, gamma=1.0):
    nA = env.action_space.n
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(nA))
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        ## TODO: complete the function
        # First set the epsilon value
        epsilon = 1.0/((i_episode/8000)+1)
        # NOTE: we can change the divisor from 8000 to a new value so that we can edit out
        # how much our epsiolon value updates.
        # NOTE: Still the same epsiode generator
        episode = generate_episode_from_Q(env, Q, epsilon, nA)
        # now we update our action-value function after the episode
        # NOTE: This is the part where the constant-alpha gets applied
        Q = update_Q_alpha(env, episode, Q, alpha, gamma)
    policy = dict((k,np.argmax(v)) for k,v in Q.items())
    return policy, Q
```

Now, Monte Carlo Methods lesson is completed. I still have to read about how the optimal decision would be reached. Also, there is more to this than just finishing the lesson. I am thinking that the next step here would be to do some OpenAI gym practice. But in the meantime, focus is still on finishing this nanodegree. I have a lot of ground to cover in terms of the lessons needed so I need to focus on those first.

Now that we have finished MCM, lets summarize some learnings first. MCM is used for agents that do not have any idea of the environment and must learn through interacting with it. Unlike DP where the agent should have an idea of all the probabilities and values of state-action pairs in order for it to come up with policy improvement to get closer to the optimal policy, MCM can make these improvements as it goes along the environment. It will constantly update and optimize its policy until it has reached the most optimal one. One pitfall of MCM is that it would likely become greedy in its policy. While this is encouraged at the later stages when it has a good grasp of the envrionment, in its early stages where it still has not done a lot of state-action pairs this behaviour should be discouraged. To resolve this greedy policy issue, we add a variable $\epsilon$ which makes the agent's actions stochastic instead of deterministic leaning towards the greedy. This is done so that the agent will do more exploring in its early stages. To balance out and reintroduce the exploitive behaviour to the agent at the later stages, we want to slowly decrease the value of $\epsilon$ so that it reaches to a point of almost nearly 0 (suggested to be 0.1 or close but never 0). Another problem with basic implementation of MCM is that it will take into account all the previous values it has for its estimates. This presents an issue because, as stated earlier, the early stages of the agent interaction with the environment is still flawed or at least not yet the general average. To fix this, we would want the agent to look more on the latest estimates and use that to decide on the policy since the latest estimates were taken when the agent already has a good sense of what the environment is. To add this to the $Q$ update, we use the constant-$\alpha$ method. The variable $\alpha$ is used to determine how recent, or how far into the past, the agent will consider when it is updating its $Q$ values.

## Day 96: October 10, 2018

Now we go to __Temporal Difference Methods (TDM)__. In MCM we were limited in updating our estimates only after the end of the episode, thus MCM is only applied to episodic task otherwise the agent will never update. What do we do then when we need a model-free approach to learning for continous task? For example, a self-driving car. If you use MCM on a self-driving car, then it would have to either crash or reach its destination first before the estimates are acheived. This would be problematic and expensive as well because we will most likely run out of cars during our initial stages when the car would tend to explore. Or for example an agent playing Chess, obviously the MCM will not be a good fit to this situation because winning or lossing in Chess is more to do with the actions in between the start and the end of the episode. This is where Temporal Difference Methods come in. From what I understand in TDM, the method will actually try to improve the estimates based on time and not based on the end of the episode. For example, in a self-driving car every time the car is able to navigate from one corner to the other without issues then the model will update its values and learn from what it did during the time it moved from the start of the block to the finish. Or in Chess, the agent will calculate the probability of winning based on the move it did and how the opponent react so that it can adjust its strategy as the game progresses.

Now on to the first Temporal Difference method we will tackle which is _one-step TD_ or _TD(0)_. Suppose we stat at $t=0$ where our state-value would have been 0, since we have no bias yet. Then we let the agent take an action and it has now taken one time interval leading us to be at $t=t+1$. At this point, we would already have a reward $R\scriptstyle_{t+1}$. From here we can already update our estimate value by factoring in the current reward plus the estimate of what the future value would be based on the current one. # NOTE: Not yet sure of this.

The first version of TD we are going to use is _one-step TD_ or _TD(0)_. The equation for the update statement would be: $V(S\scriptstyle_t)$ $\leftarrow$ $V(S\scriptstyle_t)$ + $\alpha(R\scriptstyle_{t+1}$ + $\gamma V($$S\scriptstyle_{t+1}$$)$ - $V(S\scriptstyle_t)$$)$. Now, what we would need to be able to do _TD(0)_ is the current state-value: $V(S\scriptstyle_t)$, the reward for the next state: $R\scriptstyle_{t+1}$, the next state-value: $V(S\scriptstyle_{t+1}$$)$. It is not included in the actual update function but since this is an evaluation, you should also know that the action: $A$ taken to reach the next state by following the policy $\pi$ is to be considered as well.

If we dissect the equation, we an get the its components as: the previous estimate: $V(S_t)$, and the __TD Target__: $\alpha(R_{t+1} + \gamma V(S_{t+1})$. If we destribute $\alpha$ we can get a function of: $V(S_t) \leftarrow V(S_t) + \alpha(R_{t+1} + \gamma V(S_{t+1}) - V(S_t))$. Now we can again see the effect of alpha with regards to estimate. Do note that it still looks the same as the estimate function we had earlier for Monte Carlo Methods in the constant-$\alpha$. Some notes on the advantages of TD(0) on MC Prediction. First is that it can update the value function estimate for every time step compared to the MC which can only do it at every end of episode. This makes TD applicable to both episodic and continous taks. Also, in practice, the TD prediction can converge faster than the MC prediction. This is yet to be proven and the Udacity team has gievn us example 6.2 of the book on RL to try on how to compare these things.

We would then implement TD(0), via a mini-project. A brief description of the mini-project environment: it is an OpenAI Gym environment called Cliff Walking. A brief summary of the environment first. The gridworld is of size 4x12 and there are 48 states in total (0-47) that the agent can move through. There are also 4 actions that the agent can do which corresponds to its movement, up, down, left and right. So we have $S^{+}={0,1,...,47}$ and $A = {0,1,2,3}$. We will try to solve this environment by using Temporal Difference method. First we will implement the TD prediction. The The psuedocode of TD(0) prediction is seen below.

![TD Prediction: TD(0) Pseudocode](https://d17h27t6h515a5.cloudfront.net/topher/2017/October/59dfc20c_td-prediction/td-prediction.png)

```note
The grid below shows the layout of the cliff walker environment.

[[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
 [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
 [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
 [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]]

 initial state = 36
 terminal state = 47
 cliff = 37 - 46

Then for the move list:
UP = 0
RIGHT = 1
DOWN  = 2
LEFT = 3
```

```python
# TODO: Implement the TD prediction for the project.
from collections import defaultdict, deque
import sys

def td_prediction(env, num_episodes, policy, alpha, gamma=1.0):
    # initialize empty dictionaries of floats
    V = defaultdict(float)
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        ## TODO: complete the function
        # As always, we begin with a reset
        state = env.reset()
        # Let's evaluate until it is done (infinite loop)
        while True:
            action = policy[state] 
            # Based on the current state we choose action A based on the policy.
            next_state,reward,done,info = env.step(action)
            # We then get the next_state, reward for action, terminal state or not and info from env
            V[state] = V[state] + alpha*(reward + (gamma*V[next_state])-V[state])
            # We do the update statement for TD(0)
            state = next_state
            # The equivalent of t = t+1
            # We then loop this until the terminal state in which case done = True
            if done:
                break
    return V
```

## Day 97: October 11, 2018

Now that we can evaluate the state-value $V(S)$, we can proceed now with the control portion of our method. For TD and specifically for TD(0), we will be using SARSA(0) control method. First of all we need to recall that for Temporal Difference on one step, we will need initially the State, Action and Reward for t. Then from that Reward at t we can get evaluate the value of the State and the Action for t+1. So we can get:

$$S_0 A_0 R_1 S_{1} A_{1} | R_2 S_2 A_2 | R_3 ...$$

Now we can see why its named _sarsa_. Basically, at the start it needs the first SARSA which corresponds to one whole time step from the initial time $t$ to one time-step forward $t+1$. From one whole starting set of $SARSA$ we can then use temporal difference to estimate the next reward $R_2$ for the given we follow the policy and take $A_1$. Then, it is just going to be a cycle of estimating the $R$ then from that follow the policy to choose which $S$ and $A$ to take and so on until we can have an improved policy and eventually reach our optimal policy.

![TD Control: Sarsa(0) Psuedocode](https://d17h27t6h515a5.cloudfront.net/topher/2017/October/59dfd2f8_sarsa/sarsa.png)

```python
# TODO: We shall first do the pre-requisite functions needed for sarsa
def update_Q(Q_0,Q_1,reward,alpha,gamma):
    # NOTE: Sarsa wil make use of Q values (action-values)
    '''This will return the Q-value computation'''
    return Q_0 + alpha*(reward + gamma*(Q_1)-Q_0)

def epsilon_greedy(env,Q_s,episode_num,eps=None):
    # NOTE: We will be using epsilon-greedy as part of our implementation of Sarsa
    epsilon = 1.0/episode_num
    if eps is not None:
        epsilon = eps # When we want to use constant epsilon
    policy_s = np.ones(env.nA) * epsilon / env.nA
    policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon/env.nA)
    return policy_s
```

```python
# TODO: Implement Sarsa TD control
import matplotlib.pyplot as plt
%matplotlib inline

def sarsa(env, num_episodes, alpha, gamma=1.0):
    # initialize action-value function (empty dictionary of arrays)
    Q = defaultdict(lambda: np.zeros(env.nA))
    # initialize performance monitor
    # loop over episodes
    plot_every = 100
    tmp_scores = deque(maxlen=plot_every) # Creating a double ended queue, this is new.
    scores = deque(maxlen=num_episodes) # Creates a record for the scores
    # We would need to loop until a the num_episodes is reached
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        score = 0 # set the score to 0
        state = env.reset() # again, we start the episode with a reset of the environment
        policy_s = epsilon_greedy(env,Q[state],i_episode)
        action = np.random.choice(np.arange(env.nA),p=policy_s)
        # Here the action taken is based on the probability of the action when following policy_s which is an epsilon_greedy policy.
        for t_step in np.arange(300): # Since we do not want to go into a loop, we need to limit our episodes.
            next_state,reward,done,info = env.step(action)
            # We first unpack our values based on the action
            score += reward
            # We then update our score with the reward we received from the action
            if not done:
                # NOTE: We take another step if we are not yet done. WE evaluate again using epsilon-greedy probabilities
                policy_s = epsilon_greedy(env,Q[next_state],i_episode)
                next_action = np.random.choice(np.arange(env.nA),p=policy_s)
                Q[state][action] = update_Q(Q[state][action],Q[next_state][next_action],reward,alpha,gamma)
                # The statement above is the updating of the action-value
                state = next_state
                action = next_action
                # After the update, we move one time-step forward so next_state/action becomes current.
            if done:
                Q[state][action] = update_Q(Q[state][action],0,reward,alpha,gamma)
                # We do one last update
                tmp_scores.append(score) # do final update of the score
                break
        if (i_episode % plot_every == 0):
            scores.append(np.mean(tmp_scores)) # We update our deque scores list with the average of scores
    plt.plot(np.linspace(0,num_episodes,len(scores),endpoint=False),np.asarray(scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Mean reward (over Next %d episodes)' % plot_every)
    plt.show()
    print(('Best Average Reward over %d epsidodes' % plot_every),np.max(scores))
    return Q
```

The first Control method we have for Temporal Difference would be _sarsa_ where we evaluate the next values $Q(S,A)$ via the $\epsilon$-greedy method similar to the one we used in Monte Carlo Methods. First we initialize all of our values to zero to since we want them to be equiprobable. Then we follow our initial $\epsilon$-greedy policy and choose our first _action_. From this action we can get our initial reward. Then from this we can get an estimated value, our _TD target_, which we will then use to evaluate the succeeding steps. Then similar to the MCM method, this will go on until we eventually reach our optimal policy.

The next Control method we will discuss for TD is the _Sarsamax_ method, which is also called _Q-learning_. The main difference for this is that instead of updating $Q(S_0,A_0)$ after the whole SARSA term has been completed, we actually update it before we take the next action $A_1$ leaving us with $SARS$ then update. Here is how the new equation looks like:
$$
S_0 A_0 R_1 S_1 | update\\
Q(S_0,A_0) \leftarrow Q(S_0,A_0) + \alpha(R_1 + \gamma \max_{\mathclap{a \in A}} Q(S_1,a) - Q(S_0,A_0))
$$

As we can see, instead of getting evaluating the next action-value based on the policy, we actually use the greedy method by simply choosing the maximum action-value for the state-action pair for state $S_1$. When we compare it to the vanilla _sarsa_ we can say that the vanilla is aproximating the $epsilon$-greedy policy while the _sarsamax_ or _Q-learning_ directly attempts to get the approximate optimal value.

![TD Control: Sarsamax Pseudocode](https://d17h27t6h515a5.cloudfront.net/topher/2017/October/59dff721_sarsamax/sarsamax.png)

```python
# TODO: Implement Sarsamax evaluation
def q_learning(env, num_episodes, alpha, gamma=1.0):
    # initialize empty dictionary of arrays
    '''
    This is going to evaluate Q-learning or sarsamax. Instead of following an state-action pair based on the epsilon-greedy policy, we are going to use the greedy policy.
    '''
    Q = defaultdict(lambda: np.zeros(env.nA))
    plot_every = 100
    tmp_scores = deque(maxlen=plot_every) # Creating a double ended queue, this is new.
    scores = deque(maxlen=num_episodes) # Creates a record for the scores
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        ## TODO: complete the function
        score = 0
        state = env.reset() # again, we start the episode with a reset of the environment
        while True:
            policy_s = epsilon_greedy(env,Q[state],i_episode)
            action = np.random.choice(np.arange(env.nA),p=policy_s)
            next_state,reward,done,info = env.step(action)
            score += reward
            # The same as sarsa, only the update statement is going to change
            # Instead of a state-action pair from the policy, we will get the highest action-value
            Q[state][action] = update_Q(Q[state][action],np.max(Q[next_state]),reward,alpha,gamma)
            state = next_state # We then move one time step, updating the current state.
            if done:
                tmp_scores.append(score) # do final update of the score
                break
        if (i_episode % plot_every == 0):
            scores.append(np.mean(tmp_scores)) # We update our deque scores list with the average of scores
    plt.plot(np.linspace(0,num_episodes,len(scores),endpoint=False),np.asarray(scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Mean reward (over Next %d episodes)' % plot_every)
    plt.show()
    print(('Best Average Reward over %d epsidodes' % plot_every),np.max(scores))
    return Q
```

Then, we move on to the final Control method for TD which is _Expected Sarsa_. Expected sarsa is similar to sarsamax where the only difference is in the update step of the action-values. Instead of finding the maximum action-value for the actions in state $S_{t+1}$, expected sarsa uses the _expected value_ of the next state-action pair and accounts for the probability that the agent will choose the action on the next state. The equation for _expected sarsa_ is seen below. The idea behind the summation is that expected sarsa is actually calculating the action-value not just by following the $\epsilon$-greedy policym or the greedy policy but all the possible actions it has to take. Simply said, its trying to optimize every action step taken based on the policy before taking the step.

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha(R_1 + \gamma \sum_{\mathclap{a \in A}}\pi(a|S_{t+1}) Q(S_{t+1},a) - Q(S_t,A_t))
$$

![TD Control: Expected Sarsa Pseudocode](https://d17h27t6h515a5.cloudfront.net/topher/2017/October/59dffa3d_expected-sarsa/expected-sarsa.png)

```python
# TODO: Implement Expected Sarsa evaluation
def expected_sarsa(env, num_episodes, alpha, gamma=1.0):
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(env.nA))
    plot_every = 100
    tmp_scores = deque(maxlen=plot_every) # Creating a double ended queue, this is new.
    scores = deque(maxlen=num_episodes) # Creates a record for the scores
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        ## TODO: complete the function
        score = 0
        state = env.reset() # again, we start the episode with a reset of the environment
        policy_s = epsilon_greedy(env,Q[state],i_episode)
        # NOTE: For expectedsarsa, we need to get the policy at the initial step. Then we will go over improving the policy as we go at the end of every iteration.
        while True:
            action = np.random.choice(np.arange(env.nA),p=policy_s)
            next_state,reward,done,info = env.step(action)
            score += reward
            # The same as sarsa, only the update statement is going to change
            # Instead of a state-action pair from the policy, we will get the highest action-value
            policy_s = epsilon_greedy(env,Q[next_state],i_episode, 0.005)
            Q[state][action] = update_Q(Q[state][action],np.dot(Q[next_state],policy_s),reward,alpha,gamma)
            # NOTE: We do not need to do np.sum since np.dot will return a scalar value that already accounts for the pairing of probabilities from policy_s and action-value Q.
            state = next_state # We then move one time step, updating the current state.
            if done:
                tmp_scores.append(score) # do final update of the score
                break
        if (i_episode % plot_every == 0):
            scores.append(np.mean(tmp_scores)) # We update our deque scores list with the average of scores
    plt.plot(np.linspace(0,num_episodes,len(scores),endpoint=False),np.asarray(scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Mean reward (over Next %d episodes)' % plot_every)
    plt.show()
    print(('Best Average Reward over %d epsidodes' % plot_every),np.max(scores))
    return Q
```

Below is the summary of the equations for the Control methods for temporal difference. Arrangement is _sarsa_,_sarsamax_ and _expectedsarsa_. Additional readings on the topic can be found on the textbook sections 6.4 to 6.6.

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha(R_1 + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t))\\
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha(R_1 + \gamma \max_{\mathclap{a \in A}} Q(S_1,a) - Q(S_0,A_0))\\
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha(R_1 + \gamma \sum_{\mathclap{a \in A}}\pi(a|S_{t+1}) Q(S_{t+1},a) - Q(S_t,A_t))
$$

The summary would be:

* Sarsa and Expected Sarsa are **on-policy** TD control algorithms. They both follow the same policy which is $\epsilon$-greedy. This policy is then evaluated and improved and used for the selection of the next actions.

* Sarsamax is __off-policy__ method of TD control. It simply follows greedy policy when optimizing the policy.

* There is better online performance for On-policy TD control methods than Off-policy methods.

* Expected Sarsa generally achieves better performance than vanilla Sarsa.

Another optional thing to do would be to come up with this figure during testing.

![Optional Exercise Fig. 6.4 Textbook](https://d17h27t6h515a5.cloudfront.net/topher/2017/December/5a36bc5a_screen-shot-2017-12-17-at-12.49.34-pm/screen-shot-2017-12-17-at-12.49.34-pm.png)

The lesson here is that Q-learning (sarsamax) is getting worse online performance. This is when the agent collects less reward on average per episode but if you look at the path it took it actually is the optimal path. Sarsa on the otherhand acheived great results in terms of average reward per episode but its policy is sub-optimal since it went to the "safe" path which is furthest from the cliff.

## Day 98: October 12, 2018

No new lessons for today. Focus is on completing all the mini-projects.

## Day 99: October 13, 2018

One more day till the 100. Still focused on the mini-projects. I have a few more lessons to go through then I can start on the Quadcopter project.

Okay, So I have just finished the coding for the Teporal difference lessons. Most of it came from the help of the solution manual. Added my take on what is happening based on the cheatsheet for the course.

For now, I am skipping first the OpenAI Taxi Gym mini-project. I am going first to go over the other topics in the module. First up for today: _Deep reinforcement learning_. Change of plan, the internet connection is not cooperating.

## Day 100: October 14, 2018

Okay, so now Deep Reinforcement Learning. But first a review. Most of reinforcement learning problems are framed as Markov Decision Process(MDP). MDPs are typically consisting of states $S$ and actions $A$ along with probabilities $P$, rewards $R$ and a discount factor $\gamma$. $P$ is showing the probability of an state and reward pair of any future time ($t+1$) is given as a function of the state and action at current time $t$, $P(S_{t+1},R_{t+1}|S_t,A_t)$, this characteristic is called the _Markov property_. We normally want to solve for two values: _State Value function $V(S)$_ and _Action Value function $Q(S,A)$_. State value is simply the value of the state considering its location relative to the goal or the reward. Action value is the value of taking an action in the context of completing the goal or getting the reward. __The *Goal* is to find the optimal policy $\pi^*$ which maximizes the total reward received__. Since MDPs are probabilistic in nature, this is where the discount factor $\gamma$ comes in. Since we cannot know for certain what the total reward is going to be, due to probababilistic tendencies, we rephrase the goal as: __To find the optimal policy $\pi^*$ which maximizes the total *expected* reward.__ Reinforcement Learning algorithms can be group into either __Model-based Learning (Dynamic Programming)__ or __Model-free learning__. Model-based learning as the name suggests requires that all the possible values and proabilities are given before any computation can be done. The methods in Model-based learning are _Policy Iteration_ and _Value Iteration_. To approach problems where the probabilities and values are unknown we need to use Model-free learning methods like _Monte Carlo methods_ and _Temporal-difference learning_. Model-free learning will get the approximate values and probabilities by actually interacting with its environment and learning updating the values as it gains more experience.

Deep reinforcement learning is actually the application of deep learning with reinforcement learning concepts. This would mean that most of the actions is going to occur in a _continous space_ instead of a _discrete space_. Table and dictionary mappings that we had for pairs of action and reward or action value for each state-action pairing will not scale well and will not be applicable in continous space. This is one of the first problem we will solve on DRL. We first need to generalize the approach we had for RL into something that scales and applies to continous spaces. But before that we need to actually define what a continous space is. Basically, continous means that it is not confined to a set of integer values [0,1,2,3] but could be values that can take on a _range_ of posibilities [0.0, 1.0]. Why continous and why do we need to have a method for continous values? For one, having discrete representation for tasks in the real world is infeasible. We simply cannot divide the whole world into grids with discrete values. If we do try to divide it into grids then we have to use a scale so small to give a good representation of the world and that would mean that our data for the coordinate system alone would be huge. Actions cannot also be simply using discrete values. The example for this would be a dart throwing robot. The force it applies, the angle of the joints etc. will have a distinct effect on where the dart will land on the board. Having them in discrete values will severly limit which parts of the board we can hit. So the need for continous spaces is actually needed for us to deal with the real physical world.

So for the first part, how do we use concepts in RL for continous spaces? One way would be _discretization_ (ironic isn't it?). By discretization, we are going to do minimal changes, if any, to our existing RL methods. Discretization is simply dividing our continous space, say the room, into grids. This way, our robot could know its position via its grid position. What if there are obstacles in the room? We simply block out any of the grids that have these obstacles, we call these grids with obstacles as _occupancy grid_. If you are familiar with the captchas that require you to select the images with buses or traffic lights etc. that is the same concept, so obviously we are going to have a problem due to the use of the grid. Instead of the robot avoiding the actual outline of the obstacles, it avoids the occupancy grid as a whole. What if our grid sizes our too large, then our robot could think that there might be no path from its current position to a goal due to the occupancy grids blocking the way. One solution for this is to use _non-uniform discretization_. One way of non-uniform discretization implementation is when we use disimilar step sizes in our grid so that we can limit the occupancy grids of obstacles allowing a path for our robot to be available. Another way to use non-uniform discretization is to divide the large occupancy grids into smaller grids and define the occupancy grids from the smaller grids. Do note that non-uniform discretization can lead to computation complications due to disimilar step sizes or due to increased complexity of the grids due to smaller step sizes. This can lead to computation and time penalties when evaluating values for the agent.

Now we go to exercises for creating discrete spaces, this is found on the udacity reinforcement learning repo.

```python
def create_uniform_grid(low, high, bins=(10, 10)):
    """Define a uniformly-spaced grid that can be used to discretize a space.
    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    bins : tuple
        Number of bins along each corresponding dimension.
    Returns
    -------
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    """
    grid = [np.linspace(low[A],high[A],num = bins[A]+1)[1:-1] for A in range(len(bins))]
    # First we do a list compression and use linspace to generate the list of points between low and high given the number of splits (bins).
    print("Format: (low, high)/bins => grid")
    for low,high,bins,splits in zip(low,high,bins,grid):
        print("    [{}, {} /{} => {}]".format(low,high,bins,splits))
    return grid


low = [-1.0, -5.0]
high = [1.0, 5.0]
create_uniform_grid(low, high)  # [test]
```

```python
# TODO: Create the discretize function below
def discretize(sample, grid):
    """Discretize a sample as per given grid.
    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    Returns
    -------
    discretized_sample : array_like
        A sequence of integers with the same number of dimensions as sample.
    """
    discretized_sample = list(int(np.digitize(sample, grid))for sample,grid in zip(sample,grid))
    return discretized_sample


# Test with a simple grid and some samples
grid = create_uniform_grid([-1.0, -5.0], [1.0, 5.0])
samples = np.array(
    [[-1.0 , -5.0],
     [-0.81, -4.1],
     [-0.8 , -4.0],
     [-0.5 ,  0.0],
     [ 0.2 , -1.9],
     [ 0.8 ,  4.0],
     [ 0.81,  4.1],
     [ 1.0 ,  5.0]])
discretized_samples = np.array([discretize(sample, grid) for sample in samples])
print("\nSamples:", repr(samples), sep="\n")
print("\nDiscretized samples:", repr(discretized_samples), sep="\n")
```

Okay, so there are more methods discussed in the introduction like Tile coding and approximation. For _tile coding_ it is heuristically splitting tiles into segments of two tiles to to aid in finding out the optimal value for the agent. This is usually done heurestically like for example decided by the contribution of the effects each tile makes to the value and then from there decide which tiles to further explore on. This is done until some maximumm number of splits are done or when an upper limit of episodes are reached. Think of it as the same principle in numerical methods in finding the derivative or integral of a function. We approximate the first one and then split the range into two and then approximate again and find out which part of the new split the approximate value will fall on to until finally we can get the approximate to be in a range so small that it is almost accurately representing the actual value.

Then there is the approximation portion. _This is where it gets interesting_. Obviously, by digitizing a contnous space into a discrete function we will not be able to represent it completely (unless if the function is simple enough). The model we would generate would be limited by the complexity of the function of the model we are trying to approximate. Recall what we did with tile coding where we approximate the location of the actual value by splitting the error and moving our estimates closer to where the actual value is. Does it sound familiar? This is because it is the same concept as _Gradient descent_. In gradient descent, we were trying to get the weights for each input so that we can reduce the error we get at the output and the target. What we got from the gradient descent is a set of weights that when multiplied to each nodes can give us a value that is close (to a certain degree) to our desired value. From these weights, we can actually get the approximate model of the function between the input and the output. In the context of reinforcement learing, we can use the same concept of gradient descent into creating a model for our environment. This will allow us to map the result of our inputs (in this case the action of an agent) to the output (in this case the total reward). So I think what we have just modeled was the approximate optimal policy $\pi^*$. WOW.

Okay, I might still be wrong about which value the approximation is used for, it could be the $V(s)$ or it might be $Q(S,A)$ but it still works out that we will be able to map out the continous space approximate for these values using the gradient descent method. Now that we have cleared that up we move on to what we can do with the information. First we go about Q learning, here are some interesting topics I have found [Stocks Analysis](https://www.kaggle.com/itoeiji/deep-reinforcement-learning-on-stock-data), [Trading](http://www.wildml.com/2018/02/introduction-to-learning-to-trade-with-reinforcement-learning/), [Playing DOOM](https://medium.freecodecamp.org/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8) and this guide on making use of [Deep Q-learning with Neural Networks](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0). Okay, so I have been reading about the posibilities of Deep Q-learning and it does really make sense to move deeper into this line of course. The ability to use deep neural networks and combine it with the learning ability of Artificial Intelligence makes the endeavour exciting. From playing Doom, to playing Go in AlphaGo, to OpenAI's Dota2 bot, the applications right now are exciting _but_ the implications are really what I want to work on. The fact that I can make a stock trader for this one is aligned with my goal of having a bot do my trades. Also, It does align with my decision to pivot towards AI, automation. I am torn between diving deeper into Deep Reinforcement learning and on using Deep Learning on finding ways to change something in society and I think Deep Reinforcement Learning is going to bring out the best of two worlds.

First we transform into Deep Reinforcement Learning the Monte Carlo Methods. The same is going to be said for Temporal Difference. The same concept is done except that it is adopted for the use in continous space and with the application of the _approximation_ model we discussed earlier to provide a model for the action-values.

Since I am not able to continue watching so now I have a fallback in the book. I have been reading about the summary of Part 1: Tabular Solution Methods. One of the main concepts that was stated was on how the methods like Dynamic programming, Monte Carlo Methods and Temporal Difference are not actually to be treated as distinct sets of method for learning but rather as a collection that is to be used in solving the agent's problem. The other concept is the _Generalized Policy Iteration_ which in essence is the one that seeks to make the learnings into actionable and coherent knowledge. It is in GPI where the two simultaneous processes are outlined. First is in making the value function consistent with the current policy (policy evaluation), and the other process is making the policy greedy with respect ot the current value function (policy improvement). In Generalized Policy Iteration, the two processes are alternating seeking to constantly improve and find the optimal value $v*$ and optimal policy $\pi*$. Almost _ALL_ reinforcement learning methods are well described as GPI, meaning that they let policy-evaluation and policy-improvement intearact, mutually exclusive in terms of internal details but symbiotic in relationship. The GPI will eventually, in theory, end when both the evaluation process and improvement process stabilize. This would mean that there is no longer a change produced in the value function and the policy optimization which can therefore be considered as the optimal values. There is a bit of irony with GPI. The evaluation process and improvement process are actually both competing and at the same time cooperating. They compete in a sense that a change in policy by following greedy would mean that the value estimates would get erroneous and has to change. When the value estimates get changed to evaluate the current policy then the policy would also be pulled into the correct direction, although following greedy again would lead to the evaluation to get erroneous. This goes on and on with both evaluation and optimization pulling the other into the opposite direction but in the end the goal of getting to an optimized function is actually acheived through this process. So we can say that they are also cooperating. Its quite a weird interaction but it works. Both are actually not attempting to directly acheive optimality but by interacting with one another inadvertently arrivess to the optimal solution. Its quite amazing really.

The flow of knowledge in the Deep Q Learning portion is so high. I don't know if its because I am already tired or if there is purely conceptual bombs being dropped on me constantly. I have to rewatch this portion some time, good thing the course materials would be available for 1 whole year.

## Day 101: October 15, 2018

For now we are moving on to implementation of Deep Q learning. [Here](https://keon.io/deep-q-learning/) is one resource to look at for implementation using Keras-python.

Today is an extension of the Standby day so I am obviously still up at 3:00AM. I am having some difficulty digesting the information I just received from Deep Q learning module. So I was taking some time off of the udacity lessons. Went on to LinkedIN and found some post about DSGO which I am looking forward to watching if they ever release a recorded vision of it. While browsing through my feed I saw [this article](https://www.linkedin.com/pulse/engineers-guide-artificial-intelligence-galaxy-kai-fu-lee/) titled: An Engineer's Guide to the Artificial Intelligence Galaxy. It really resonated with me. Although I am not yet inside the field of AI, I am already making the pivot towards it and I am excited but also scared.

Here are some of the points Mr. Kai Fu Lee adviced so as we can have the time of our life in the comming decade: one is:

> Embrace AI, and align your career by betting on its inevitability.

Yes, change led by AI will eventually transform the way we live much like the industrial revolution and the introduction of electricity. Its normal to be fearful of the unknown. By knowing that you are good enough and that you have been trained for the changes you have worked hard to overcome. While it might make you feel uneasy, feeling like you are not in control, like you are not in the direction you wanted, remember to embrace the changes and look at the positive things that you will make of it. We must warmly embrace Ai, and while it may fail at our times just like how things are on the alpha and beta versions, we must, for the sake of its improvement, adapat to it and be prepared to catch its mistakes for some time. Given more time to learn, it will eventually get better and that will lead to the overall good. It just needs data and some ELS and then when its stable, it will just get better.

Here is an idea, what if we link together an AI that will invest in the market and manage money and an AI that will classify loan takers. Has that been done before? I think that would work right? The trader AI will actively create the wealth and the loan-AI will check out who we can loan money to. Imagine if this was done in 4WD, I think if there is data on the rate of default for loans under certain conditions, we can easily underwrite some loans to the needy and deserving. One makes the money, the other distributes it. The money will circulate but it also would accumulate. Responsible money making.

## Day 102: October 16, 2018

For now, reading on the paper of DQN. The premise is that the researchers were able to create a system that was able to play an Atari2600 game with the use of Q-learning algorithms paired with Deep Learning techniques. The result was that the agent was able to learn how to play the games purely based on the interaction with the environment. In the end, the agent was able to play and often surpassed human-level scores for most of the games and improved upon the best linear learner for the games it played. So how did they do it? They actually used deep neural networks as a way to create the model for the agent. The result was that the model for the agent became closer to the approximate and became a vectore instead of a simple linear model. What happened was $V(s)$ became $\hat{V}(s,w)$ where $w$ is the new weight array. So in terms of deep neural nets, the input is the state and the model becomes the weight and the output becomes the actions. The values were corrected based on the return. In Q-learning, there is a function called Q-function, this is used to approximate the reward based on a state. We call this $Q(s,a)$ where $Q$ calculates the expected future value given state $s$ and action $a$. Recall from Deep Neural Networks that our network will need two values for it to be able to get the error and apply gradient descent to adjust the weights. So we need to define the _target_ and _prediction_ fomulas. Both our target and predictions are obviously the Q-functions. For our target it could be as simple as $r+\gamma \max \hat{Q}(s,a)$ and for our prediction its going to be $\hat{Q}(s,a)$. From this we can get the loss as $(r+\gamma \max \hat{Q}(s,a) - \hat{Q}(s,a))^2$.

Had to stop for a bit since there was a critical issue at work. All hands on deck for this one.

Okay, so I have resumed watching the remaining RL methods. I have just finished the _Policy-based methods_ and I am now on _Actor Critic method_. Right now the concepts are a bit hazy for me since there is no examples given yet, it has been mostly math equations and derivations and context on why it works. I have to do some searches for more digestible explanation since the videos is a bit lacking for me.

Now I have completely watched over the entire lessons. The idea now is to make sure that I can take on the project which seems like a capstone to be honest. It is a bit daunting for now but I know I'll get there. I have 12 days to complete the task. So for now, its on to planning first. Right now I have just downloaded a copy of the notebook in the workspace. The expected outcome is not that the quadcopter is to fly flawlessly but for the resulting data to show that the algorithm is actually converging.

So the plan later would be to read up on DDPG, possibly implement it to simple tasks first, OpenAI Taxi possibly. Then from there get the basic gists of it and reframe that to the problem in Quadcopter.

## Day 103: October 17, 2018

[Deep Reinforcement Learning](
    
) syllabus. Post of implementation of [Deep Deterministic Policy Gradinents in Tensorflow](https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html) by Patrick Emami. Using Keras and DDPG [article](https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html).

So from [this article](https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html) I am learning about what _Deep Deterministic Policy Gradient (DDPG)_ is. Since the inception of DQN, researchers were able to realize that deep learning methods can now be.

![actor-critic architecture from Sutton's book](https://yanpanlau.github.io/img/torcs/sutton-ac.png)

One thing I need to resolve is my AWS account. It would not be possible to deploy and train this on my PC.

```python
class TaxiEnv(discrete.DiscreteEnv):
    """
    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich
    Description:
    There are four designated locations in the grid world indicated by R(ed), B(lue), G(reen), and Y(ellow). When the episode starts, the taxi starts off at a random square and the passenger is at a random location. The taxi drive to the passenger's location, pick up the passenger, drive to the passenger's destination (another one of the four specified locations), and then drop off the passenger. Once the passenger is dropped off, the episode ends.
    Observations: 
    There are 500 discrete actions since there are 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger is the taxi), and 4 destination locations. 
    
    Actions: 
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east 
    - 3: move west 
    - 4: pickup passenger
    - 5: dropoff passenger
    
    Rewards: 
    There is a reward of -1 for each action and an additional reward of +20 for delievering the passenger. There is a reward of -10 for executing actions "pickup" and "dropoff" illegally.
    
    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters: locations
    """
```

A big change, career wise, has been decided today. Actually, it was never decided today but has started out as an idea some days prior to the start of this pledge. Today I have made clear my intentions. This is why its actually so liberating. Now that it is decided, all the more reason to focus and give this all that I have.

## Day 104: October 18, 2018

Okay, so it is now the 18th. I plan to at get a grasp of Actor Critic implementation and use it on Cart Pole environment in OpenAi Gym just so that I can guage if I am able to make it converge. From there we can start transitioning to the harder part of coding our Actor Critic method for the quadcopter in the project.

For now, I am trying to understand _Deep Q learning_ via this [article](https://medium.freecodecamp.org/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682) on freecodecamp.

![Architecture DDQN](https://cdn-images-1.medium.com/max/1000/1*FkHqwA2eSGixdS-3dvVoMA.png)

[Actor Critic Method AC2 with sonic hedgehog](https://medium.freecodecamp.org/an-intro-to-advantage-actor-critic-methods-lets-play-sonic-the-hedgehog-86d6240171d). This article is about the theory behind A2C.

I am currently doing DQN and applying it to CartPole env. So far, my model is not converging. I still have a long way to go before the A2C model can be tackled. So much pressure for now. But we have to push on. I am going to try using Kaggle to take this on. Training is progressing slowly on my laptop (no surprises there). Sidenote: They are constantly hammering something on the walls somewhere near my unit, its annoying. :annoyed:

```python
#NOTE:
'''
The source of this is the article explaining DQN: https://keon.io/deep-q-learning/.
They have a github repo with the compiled codes which we are using as reference to code the Cart Pole DQN agent.
'''

import gym
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import h5py
Episodes = 100 # Maximum training episodes
#TODO: Define te clss of the DQN agent
class dqn_agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size # Get the state size: Number of states
        self.action_size = action_size
        self.memory = deque(maxlen=2000) # double ended list with maximum elements of 2000
        self.gamma = 0.618 # Discount rate for future rewards
        self.epsilon = 1.0 # max exploration coefficient
        self.epsilon_min = 0.01 # minimum exploration coefficient
        # self.epsilon_decay = 0.995 # we will use exponential decay for epsilon
        self.epsilon_decay = 0.02 # we will use exponential decay for epsilon
        self.learning_rate = 0.001 # For Adam optimizer
        self.model = self._build_model()
        self.epsilon_value =[]

    def _build_model(self):
        # TODO: Create the neural network with keras
        model = Sequential()
        model.add(Dense(24, input_dim = self.state_size, activation = 'relu'))
        model.add(Dense(24, activation = 'relu'))
        model.add(Dense(self.action_size,activation = 'linear')) # To choose action
        model.compile(loss='mse', optimizer = Adam(lr = self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        # Creating the memory buffer for replay

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # choose a random action Explore
        # Predict the reward value given the state
        act_values = self.model.predict(state)
        # Pick the action to take that maximizes reward
        return np.argmax(act_values[0])
    # def replay(self, batch_size):
    #         minibatch = random.sample(self.memory, batch_size)
    #         for state, action, reward, next_state, done in minibatch:
    #             target = reward
    #             if not done:
    #                 target = (reward + self.gamma *
    #                         np.amax(self.model.predict(next_state)[0]))
    #             target_f = self.model.predict(state)
    #             target_f[0][action] = target
    #             self.model.fit(state, target_f, epochs=1, verbose=0)
    #         if self.epsilon > self.epsilon_min:
    #             self.epsilon *= self.epsilon_decay
    def replay(self, batch_size):
        minibatch = random.sample(self.memory,batch_size) # Retrieve a minibatch
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done: # evaluate
                target = (reward + self.gamma*np.amax(self.model.predict(next_state)[0]))
                target_f = self.model.predict(state)
                target_f[0][action] = target
                self.model.fit(state,target_f,epochs=1,verbose = 0)
            if self.epsilon > self.epsilon_min: #Introduce decay
                self.epsilon = self.epsilon_min + (1-self.epsilon_min)*np.exp(-self.epsilon_decay*episode)
                # self.epsilon *= self.epsilon_decay
            # self.epsilon_value.append((self.epsilon,episode))
    def load(self, name):
        self.model.load_weights(name)
    def save(self,name):
        self.model.save_weights(name)

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = dqn_agent(state_size,action_size)
    done = False
    batch_size = 32
    reward_list = [] # for plotting rewards later

    for episode in range(Episodes):
        state = env.reset()
        state = np.reshape(state,[1,state_size])
        for time in range(500): # Max steps per episode
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action) # unpack the interaction
            reward = reward if not done else -10 # add reward until it falls then subtract 10 to total
            next_state = np.reshape(next_state,[1,state_size])
            agent.remember(state,action,reward,next_state,done)
            state = next_state
            if done:
                print('episode: {}/{}, score: {},epsilon:{:.2}'.format(episode,Episodes,time,agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size, episode)
        reward_list.append((episode,time,agent.epsilon))
        # if episode % 100 == 0: # Save every 100 episodes
        #     agent.save("./save/cartpole-dqn.h5")

#TODO: Create the graph to plot the training progress
import matplotlib.pyplot as plt
%matplotlib inline
def running_mean(x,N):
    cum_sum = np.cumsum(np.insert(x,0,0))
    # N here would be the range of the the moving average
    return (cum_sum[N:] - cum_sum[:-N]/N)

eps, reward, epsilon= np.array(reward_list).T
moving_ave_reward = running_mean(reward_list,10)
plt.plot(eps[-len(moving_ave_reward):],moving_ave_reward)
plt.plt(eps,reward,epsilon color = 'grey', alpha = 0.3 )
plt.xlabel('Episode')
plt.ylabel('Reward')
# NOTE:
# Expected value is that the moving averge should at least be increasing.
# Epsilon is supposed to decay.
```

Okay, so I am now on my first run for the agent. Based on the current results at 250/500, it is still erratic in its steps. I don't think it is learning that well. The scores are still in the 10 to 20 region. I still have to check it but it looks like the actions are still random. I am not sure if I have to edit it out yet, I am currently waiting for the run to finish so that we can gauge how it did but for now that is what I am thinking is wrong. Or possibly the discount factor $\gamma$ is too high at 0.95, I'll change it later to 0.618. This is what is hard with DL networks, its hard to see the effect of hyperparameters to the training.

Its funny, I figured out why it was not convering. I made an error in the indention and it affected the updating of the $\epsilon$ value. That was why the epsilon seemed to get smaller but the training is stuck. It was constantly choosing random actions because it thought that it was still in the exploration phase. :smiling_imp: So for now, it is training and it is converging. I was right when I said that there would be a cap on what the maximum possible reward would be, its 500 but for some reason the results in the reference showed a value of 1000+.

So now, it is converging. For this environment, it seems that a high $\gamma$ is still considered and is still bearable compute wise. Its effects show in the results, the higher the gamma, the better total score would be obviously. The effect is also that it would be able to acheive the higher score faster than a low $\gamma$ value, which I did not know initially. The things we learn when we actually try and experiment is great.

Next up would possibly be an $atari$ game using actor critic.

## Day 105: October 19, 2018

Trying out Kaggle for GYM. If it works, then I have a faster method of training since it would allow the training on a GPU. So we are able to run the gym inside Kaggle. Although it does not have the render option. So we might have to use AWS still. For now, I am slowly getting the grasp of DQN which is basically a part of DDPG. From what I understand, DDPG is simply two DQNs separately training from one another but are merged to provide feedbacks and improve the policy. Think of it as a Generative Adverserial Network of some sort. The actor is the generator, trying to provide the best performance or the best policy. Then you have the critic or the discriminator, its job is to comment on the performance of the actor to make sure that it improves at the same time tries to improve itself on how to give out relevant comments. Together, while they are adverserial, they make the process of policy improvement work.

TODO: Search for simple implementation of Actor-Critic later. Might have to do some downloading again on Sunday. DDQN etc.

## Day 106: October 20, 2018

There will be minimal work done today. I have a seminar to attend to for electronics engineering. Time to check out what would be the changes in the comming years with regards to the ECE in Philippine context.

For today's topic, we first had obviously the keynote: which for some reason was not coherent. It was not really driving home a point that we can act upon or make us thing. He just stated the current state of ECE in the Philippines in some mix and match of examples all diverging instead of convergin to a single point.

We had a speaker for biomedical engineering and how ECE can make a contribution with regards to management and upkeep of our Technological assets in terms of Biomedicine and what the current difficulties he noticed from his conduction of seminars all over the country. Then we had AGILE project management which was related to my current track. Its simply sumarized the points we need to look out for and cover in terms of handling projects, be it personal or professional and in a scalable way. "We are all Project Engineers" he said, pointing out the fact that even in our own personal life we can apply the concepts of AGILE to constantly evolve and deliver. Then we had a talk on the cognizance of Solar Power in the Philippine context. Mostly it was a re-introduction of solar power concepts and PV. The new things learned was actually on the calculations and installations of the panels as well as the computations of ROI for installations.

For the afternoon session it was more tightly packed. Skipped the talk on 5G, 2016 there was already a talk about 5G presented by an IEEE felow in my university. Now, in the local it is still the same thing and its 2018. The same concepts regarding the needed improvements in 5G: A better control solution for the transmitters so that we can save up on power consumption as well as more cell penetrations for densely packed environments to catch up with the demand due to the rise of Data and mobile devices which is projected to increase further. Returned back to the venue and caught the second half of Avionics talk from a PECE that was on the MRO hub. Listened to him discuss on the different panels and electronics inside the aircraft. He also discussed briefly the redundancies in the power sources/supply for the aircraft. Then we had a short break followed by a back-to-back talk on Bonding and Grounding as well as Data Center design. We discussed the different standards and practices with regards to DC design as well as the code for proper cabling and grounding and bonding for design. There was still another talk with regards to FO splicing but that was already way past the program time of 4:15PM. We called it a day by then, it was already 6:00PM. Very clear example of just how "professional" engineers are. No wonder we keep delaying projects, we can't even comply to time budgets and constraints.

## Day 107: October 21, 2018

Reading again A2C and A3C methods. Have to catch up on the implementation of the system for the project. I might have to take some more days off just o finish within time for the project. Found a very useful [repo for A2C in CartPole](https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/4-actor-critic/cartpole_a2c.py). It was good in a sense that it gives some comments with regards to the code blocks.

## Day 108: October 22, 2018

Finishing up on the A2C CartPole agent so that we can do the training.

## Resources

[Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/)

[Books Source](https://www.manning.com/)

[CS231N lecture notes](http://cs231n.github.io/convolutional-networks/)

[The 9 Deep Learning Papers You Need To Know About](https://adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html)

[Elite Data Science ML Projects for beginners](https://elitedatascience.com/machine-learning-projects-for-beginners)

[Elite Data Science: Becoming a Data Scientist](https://elitedatascience.com/become-a-data-scientist)

[Github repository for Implementations from Barto et.al. Book](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)

## For Projects

[Viber Build a Bot](https://github.com/Viber/build-a-bot-with-zero-coding)

[100 Best Github Chatbot](http://meta-guide.com/software-meta-guide/100-best-github-chat-bot)

[Viber Bot with Python](https://github.com/Viber/viber-bot-python)

[ChatterBot](https://github.com/gunthercox/ChatterBot)

[Fast.ai season 1 episode 22: Dog Breed Classification](https://towardsdatascience.com/fast-ai-season-1-episode-2-2-dog-breed-classification-5555c0337d60)

[8 Machine Learning projects for beginners](https://elitedatascience.com/machine-learning-projects-for-beginners)

[ML-AI Case studies](https://towardsdatascience.com/september-edition-machine-learning-case-studies-a3a61dc94f23)

[WildML's 2017 AI and Deep Learning year end review](http://www.wildml.com/2017/12/ai-and-deep-learning-in-2017-a-year-in-review/)

[PSEi data](https://www.johndeoresearch.com/data/)

[A post on PSEGet](http://pinoystocktrader.blogspot.com/2010/11/amibroker-charting-software-chart-data.html)

[Carto - Location and Data](https://carto.com/)

[Stock trader using DDPG](https://towardsdatascience.com/a-blundering-guide-to-making-a-deep-actor-critic-bot-for-stock-trading-c3591f7e29c2)