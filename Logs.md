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
* [x ]Setup Anaconda correctly (uhmm) :smile:
* [x]Provide a copy of the Lesson2 Jupyter notebook to the repo.

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

So I am now in CNN, and I have just finished 7 videos. Learned on how images are interpreted by computers: basically as a huge tensor with a base element of a pixel. Then we went on to discuss MLPs and how they are used for image classification and where they are trumped by CNNs: the explanation was that CNNs are more suited for multi-dimensional analysis where it looks for coreelation not just in value but also in the relative position of the elements which obviously works well with images. Then we went on with Categorical Cross-Entropy for the loss function and how it is going to be used in the context of identifying an image: Basically, the model will output the probabilities of the labels and the error is taken from those probabilities taken together and compared with the probabilities of the one-hot encoded label. Then we moved on to validating models in Keras: there was an article about the MNIST data set and how it came to be and also about previous researches done on the data set and its results. Also was able to read more on the Keras documentation, I remember it was the __callback__ class where we get to store data of our training runs and see how our model is proceeding with its training. Based on the documentation on the __callback__ class there were also some interesting functions like _earlystopping_ which stops the training when the loss or accuracy is not improving and _adjusting LR on plateu_ where the LR is decreased automatically when a patience epoch threshold is met to ensure learning progresses.<br>
That is all for now, will read more on this blog from [Machine Learning Mastery](https://machinelearningmastery.com/about/). I will find a way to download the opencv-python package later.

## Day 28: August 3, 2018

So, I was able to finally download the opencv-python package for the aind2 course. The plan now is to play around with the values in the network and go over the notebook to come up with the way the model was built. __:yum:__ <br>
Now the training and testing begins on the model. This is just Keras so nothing fancy, the objective here is to figure out where overfitting starts. __*Overfitting happens when the validation loss is higher than (by a significant ammount) the training loss*__. Here is an [interesting read](https://stackoverflow.com/questions/2976452/whats-is-the-difference-between-train-validation-and-test-set-in-neural-netwo) on the implications of validation sets on overfitting. We have known about data set spilts from our Introduction to Neural Networks, I believe its a Machine Learning concept or even an AI concept. But the idea is that we do not just burn through all our data in training, we have to have an idea of how well our model is able to predict an output or label from a data set that it has not yet seen before (think of it as the blind test). Depending on where you read, they say a good split is 20-test then 80-train, or 10-test and 90-train. The idea is that you want as much data as you can to train your model but have enough remaining data set to be able to test your model. This time we are adding another split to the __traininng data__ which is called the __*validation set*__. The validation set is usually 20% or so of your training data. The idea of the validation set is that it allows you to guage the tendency of your current model to overfit. Validation testing is done __while training__. In a way its like testing your data before hand, after each epoch or some epochs, to ensure that the increase in weights is actually going to contribute to the increase in accuracy of the model as a whole.<br>
> When your training set increases its accuracy more and more after every pass but your validation tests are the same then the model is not actually learning anything new but simply memorizing the training set. This is a __sign of overfitting__.<br>
When overfitting is detected the training should perform an early stop so that the model does not overfit the training data.<br>
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

#### Notes on TensorFlow CNN functions

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

[Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/)

[Books Source](https://www.manning.com/)

[CS231N lecture notes](http://cs231n.github.io/convolutional-networks/)