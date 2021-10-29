import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTENC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import ADASYN
import pickle
#import data
df_train = pd.read_csv('/Users/jaadeoye/Desktop/MTP_surv4.csv')
features = ['P1','P2', 'P3', 'P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14',
                 'P15','P16','P17','P18','P19','P20','P21','P22','P23','P24',
              'P25','P26']
x = df_train[features]
y = df_train.MTP
#train model
sm = ADASYN(random_state=0)
x_res, y_res = sm.fit_resample(x,y)
dt = DecisionTreeClassifier(max_depth = 2, random_state=0)
dt.fit(x_res,y_res)
pickle.dump(dt, open('dt', 'wb'))
