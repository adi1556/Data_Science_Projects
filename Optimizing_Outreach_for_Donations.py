import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import svm

# #############################################################################
#datasets
train_data = "train.csv"
test_data = "test.csv"
cols_train_r = ['id','title','zip','dob','avggift','lastgift','ramntall','ngiftall','cardgift','minramnt','minrdate','maxramnt','maxrdate','lastdate','fistdate','numprm12','cardpm12','numprom','maxadate','cardprom','domain','responded','amount']
cols_train = ['id','title','dob','zip','avggift','lastgift','ramntall','ngiftall','cardgift','minramnt','minrdate','maxramnt','maxrdate','lastdate','fistdate','numprm12','cardpm12','numprom','maxadate','cardprom','domain','responded']
cols_test = ['id','title','dob','zip','avggift','lastgift','ramntall','ngiftall','cardgift','minramnt','minrdate','maxramnt','maxrdate','lastdate','fistdate','numprm12','cardpm12','numprom','maxadate','cardprom','domain']

ddf_train = pd.read_csv(train_data, usecols=cols_train)
ddf_train_r = pd.read_csv(train_data, usecols=cols_train_r)
ddf_test = pd.read_csv(test_data, usecols=cols_test)


# Concatenate all data into one DataFrame
ddf_train["zip"]=ddf_train["zip"].replace('\-','',regex=True).astype(int)
ddf_train_r["zip"]=ddf_train_r["zip"].replace('\-','',regex=True).astype(int)
ddf_test["zip"]=ddf_test["zip"].replace('\-','',regex=True).astype(int)

s = pd.get_dummies(ddf_train["domain"])
ddf = ddf_train.drop(['domain'],axis=1)
ddf = ddf.join(s)

X_train = ddf
y_train = ddf['responded']
X_train = X_train.drop(['responded'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.67, random_state=0)

#Classification model for predicting will they respond or not
svr = ensemble.GradientBoostingClassifier(n_estimators=100)
svr.fit(X_train, y_train)
result = svr.score(X_test, y_test)
print(result)

s = pd.get_dummies(ddf_test["domain"])
ddf = ddf_test.drop(['domain'],axis=1)
ddf = ddf.join(s)
X_test = ddf
ddf_responded = svr.predict(X_test)
count =0
for i in ddf_responded:
    if i == 1:
        count = count + 1
print(count)
X_test["responded"] = pd.DataFrame(ddf_responded)
print(X_test)

#X_test.join(ddf_responded)
X_test = X_test[X_test['responded'] == 1]
print(ddf_train_r['amount'])
x = ddf_train_r['amount'].fillna(0)
X_train['amount']=pd.DataFrame(x)
#X_train['amount'].fillna(0)
ddf_train = pd.read_csv(train_data)
ddf_train["zip"]=ddf_train["zip"].replace('\-','',regex=True).astype(int)
y = ddf_train['responded']
X_train["responded"] = pd.DataFrame(y)
#X_train.join(y)
X_train = X_train[X_train['responded'] == 1]
y_train = X_train['amount']
X_train = X_train.drop(['amount'],axis=1)

#Regression
params = {'n_estimators': 100, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params) #0.0475
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(y_pred)

zip_cost = pd.read_csv("zipCodeMarketingCosts.csv")
test = pd.read_csv(test_data)
amount = 0.0
sum1 = 0.0
j = 0
for i in range(len(y_pred)):
    value = 0.0
    zipcode = X_test['zip'].iloc[i]
    cost = zip_cost.loc[zip_cost['zip'] == zipcode,'marketingCost'].iloc[0]
    sum1 = sum1 + y_pred[i]
    value = y_pred[i] - cost
    #print(value)
    iid = X_test['id'].iloc[i]
    print(iid)
    if value > 0.0:
        j = j+1
        amount = amount + value
        #new = test[['id', 'market']].set_index('id')
        #print(new)
        #new.update(test.set_index('id'))
        #test.market[iid] = 1
        #test['market'].iloc[iid] = 1
        #if test['id'] == iid:
        a = test.id[test.id == iid].index.tolist()
        print(a[0])
        test.set_value(a[0], 'market', 1)
        #print(new)
        #test['market'].new = 1
print(j)
print(sum1)
print(amount)

#test.to_csv("outpur.csv")
