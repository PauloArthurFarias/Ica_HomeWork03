# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
reduced_grant_set = pd.read_csv('reducedSet.csv')
reduced_grant_set.shape
testing_set = pd.read_csv('testing.csv')
testing_set.shape
training_set = pd.read_csv('training.csv')
training_set.shape

# %%
selected_predictors = reduced_grant_set.iloc[:, 0].tolist()
x_train = training_set[selected_predictors]
y_train = training_set['Class']
x_test = testing_set[selected_predictors]
y_test = testing_set['Class']
x_train.shape, y_train.shape, x_test.shape, y_test.shape

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ConfusionMatrix

LR_model = LogisticRegression(random_state=1, max_iter=1000)
LR_model.fit(x_train, y_train)
predict = LR_model.predict(x_test)
accuracy_score(predict, y_test)

cm = ConfusionMatrix(LR_model)
cm.fit(x_train, y_train)
cm.score(x_test, y_test)