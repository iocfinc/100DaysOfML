"""
This is the example code in SKlearn.

To better understand it, let's make up a story.
For example we want to create a classifier to test whether a person
is a drug user  or not. To do this we gather 2 parameters from the patient.
One is the color of their urine from pale to orange (0 to 2).
The other is the concentration of a substance in their blood (-10% to 20%).

The X values are the Training data. Its what we have gathered during tests.
X1 is the Urine color and X2 is the substance concentration.

The Y values are the Target data. We label each of the results as "User" or
"Non-User" from our initial tests.

With X and Y values as input, we then "train" our classifier by giving it
the data and the label. The module will take care of the training for us.

"""
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# X is the Training vectors.
Y = np.array(['Boy', 'Boy', 'Boy', 'Girl', 'Girl', 'Girl'])
# Y is the Target.


# TODO: Fix up the story 
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, Y)
print(clf.predict([[-0.8, -1]]))

clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))

print(clf_pf.predict([[-0.8, -1]]))