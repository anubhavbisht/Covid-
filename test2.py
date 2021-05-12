# training using random forest classfier
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import pickle

patientdata = pd.read_csv("data.csv")


def splitpatientdata(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    testsetsize = int(len(data)*ratio)
    testindices = shuffled[:testsetsize]
    trainindices = shuffled[testsetsize:]
    return data.iloc[trainindices], data.iloc[testindices]


train, test = splitpatientdata(patientdata, 0.2)

X_train = train[['fever', 'bodypain', 'age',
                 'runnynose', 'diffbreath']].to_numpy()
X_test = test[['fever', 'bodypain', 'age',
               'runnynose', 'diffbreath']].to_numpy()

Y_train = train[['infectionprob']].to_numpy()
Y_test = test[['infectionprob']].to_numpy()

rfc_b = RFC()
rfc_b.fit(X_train, Y_train)
y_pred = rfc_b.predict(X_train)
print('Train accuracy score:', accuracy_score(Y_train, y_pred))
print('Test accuracy score:', accuracy_score(Y_test, rfc_b.predict(X_test)))

