
# Lesson 3: Anacondas

Moving on to the next lesson, we have an introduction to Anacondas. In the introduction video we get an explanation on why we use Anaconda. One thing that struck me was that Anaconda is used since it has pre-loaded libraries that are useful. The other reason was that it allows us to make use of the Conda environment manager. Now why do we need an environment manager? The explanation is that so we can control the version of our modules and our interpreter. One possible upside here is that we can be sure that all our modules actually work together. The other thing is that we can add the portability of our code through sharing by actually having all the dependencies for our project in one virtual environment. Well now it makes sense to me, before in Pycharm I actually just use one single virtual environment, the one with the most modules and if there are missing modules I just add them. Knowing the importance of environment control, I now appreciate its value and its utility.

Why anaconda? Anaconda has functionalities that make it easier to get started in data science. It comes pre-loaded with packages that are usually used for data science like numpy, pandas and sklearn. With Anaconda there is also a package and environment manager that comes with it called __conda__. Again as I have found out, the utility of being able to manage your environment per project allows us less stress in dealing with compatibility issues or allows us to use an older version of a library that might have a functionality that is no longer available on newer versions.

### Package Managers: conda and pip

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

### Environment managers: conda

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

## Practicing the environment commands for conda

### Listing available environments (_conda env list_)

![2018_07_12_01_38_54_Administrator_Anaconda_Prompt.png](attachment:2018_07_12_01_38_54_Administrator_Anaconda_Prompt.png)

### Changing to a different environment

![2018_07_12_01_42_43_DL_ND_Lesson_3.png](attachment:2018_07_12_01_42_43_DL_ND_Lesson_3.png)

### Checking the packages in the environment

![2018_07_12_01_44_31_DL_ND_Lesson_3.png](attachment:2018_07_12_01_44_31_DL_ND_Lesson_3.png)

### Creating a .yaml file for sharing environment settings

![2018_07_12_01_45_11_Administrator_Anaconda_Prompt.png](attachment:2018_07_12_01_45_11_Administrator_Anaconda_Prompt.png)

### Checking the .yaml file

![2018_07_12_01_47_44_Fernando_Ira_Oliver_Catacte.png](attachment:2018_07_12_01_47_44_Fernando_Ira_Oliver_Catacte.png)

Take note that the .yaml file will be created in the current directory conda terminal was in.


## Best practices

### On using environments

It is good practice to setup a generic environment for the 2 versions of Python. One for Python 2 and one for Python 3. This allows you to have a general purpose environment for testing. You can then populate the environment with the basic packages for the project you want. Or you can keep it generic if you just want to practice.

### On sharing environments

When we upload or share our work and projects to others, for example in github, it is good to include a requirements.yaml file or a requirements.txt file to help users replicate the working environment you had. The requirements.txt file can be obtained via [_pip freeze_](https://pip.pypa.io/en/stable/reference/pip_freeze/) and can be handy because not everyone is using conda. 

## Further Readings for the Topic on Condas

[Conda: Myths and Misconceptions](https://jakevdp.github.io/blog/2016/08/25/conda-myths-and-misconceptions/)

[Conda documentation](https://conda.io/docs/user-guide/tasks/index.html)

