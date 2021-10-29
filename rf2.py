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
from sklearn.ensemble import GradientBoostingClassifier
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
rf = RandomForestClassifier(max_depth = 1, random_state=0, class_weight ='balanced')
rf.fit(x,y)
pickle.dump(rf, open('rf', 'wb'))
