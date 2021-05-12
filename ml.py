from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import pickle
# first read data from csv file
patientdata = pd.read_csv("data.csv")

# train test split data for applying machine learning model


def splitpatientdata(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    testsetsize = int(len(data)*ratio)
    testindices = shuffled[:testsetsize]
    trainindices = shuffled[testsetsize:]
    return data.iloc[trainindices], data.iloc[testindices]

# 20% of patientdata will go to test set data(which will be used to test your ML output)
# and 80% of patientdata will go to train set data
# which will be used for making the ML model


train, test = splitpatientdata(patientdata, 0.2)
# Now start making ML Model

# Now we will remove infectionprob from train set and test set as
# we want to apply ML model on the data and we dont result(infectionprob)
# already there

X_train = train[['fever', 'bodypain', 'age',
                 'runnynose', 'diffbreath']].to_numpy()
X_test = test[['fever', 'bodypain', 'age',
               'runnynose', 'diffbreath']].to_numpy()
# Now we want a seperate data set of result(infectionprob)
Y_train = train[['infectionprob']].to_numpy()
Y_test = test[['infectionprob']].to_numpy()
# Apply ML model in this project we are using RandomForestClassifier to
# estimate output

model = RandomForestClassifier(n_estimators=100,
                               bootstrap=True,
                               max_features='sqrt')
# now we will fit model on the above X_train and Y_train
model.fit(X_train, Y_train)
inputfeatures = [100, 1, 75, 1, 0]
infectionprobabilty = model.predict_proba([inputfeatures])[0][1]
print(infectionprobabilty)
# open a file where you want to store data
file = open('model.pkl', 'wb')
# dump information to that file
pickle.dump(model, file)
file.close()
