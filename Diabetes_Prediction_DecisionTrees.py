import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
from sklearn.preprocessing import scale
import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
print("Packages LOADED")

import os

print(os.getcwd())
os.chdir('E:\\Locker\\Sai\\SaiHCourseNait\\DecBtch\\R_Datasets\\')
print(os.getcwd())

data = pd.read_csv('diabetes2.csv')

data.info()


import matplotlib.pyplot as plt
import seaborn as sns
import re
from IPython.display import Image
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.externals.six import StringIO

from sklearn.ensemble import RandomForestClassifier,
RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error


from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
pd.set_option('display.notebook_repr_html', False)
plt.style.use('seaborn-white')
print("Package Loaded")

array = data.values
array
type(array)

X = array[:,0:8] 
X

y = array[:,8] 

y

test_size = 0.33
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size)
print('Partitioning Done!')

from sklearn import metrics

model =
RandomForestClassifier(max_depth=5,n_estimators=500,oob_score=True)

model.fit(X_train,y_train)

prediction = model.predict(X_test)
prediction

outcome = y_test

print(metrics.accuracy_score(outcome,prediction))

print(metrics.confusion_matrix(y_test,prediction)) #
