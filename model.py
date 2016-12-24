import csv
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import StratifiedKFold

np.random.seed(0)


# reading data
def read_data(file_name):
	f = open(file_name)
	f.readline()
	samples = []
	target = []
	for line in f:
		line = line.strip().split(",")
		sample = [float(x) for x in line]
		samples.append(sample)
	return samples

#load data
def load():
	print "Loading data..."
	filename_train = 'train.csv'
	filename_test = 'test.csv'
	train = read_data("train.csv")
	y_train = np.array([x[0] for x in train])
	X_train = np.array([x[1:] for x in train])
	X_test = np.array(read_data("test.csv"))
	return X_train, y_train, X_test


X, y, X_sub = load()


shuffle = False
n_folds = 10

if shuffle:
	idx = np.random.permutation(y.size)
	X = X[idx]
	y = y[idx]

#classifiers used
clfs = [
	RandomForestClassifier(n_estimators=100,max_features='auto',max_depth=17,min_samples_split=1,criterion='gini',n_jobs=-1),
	RandomForestClassifier(n_estimators=150,max_features='auto',max_depth=17,min_samples_split=1,criterion='entropy',n_jobs=-1),
	ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
	ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
	GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)
	]

skf = list(StratifiedKFold(y, n_folds))

dataset_blend_train = np.zeros((X.shape[0],len(clfs)))
dataset_blend_test = np.zeros((X_sub.shape[0],len(clfs)))

print 'X_test.shape = %s' % (str(X_sub.shape))
print 'Train.shape = %s' % (str(dataset_blend_train.shape))
print 'Test.shape = %s' % (str(dataset_blend_test.shape))

for j, clf in enumerate(clfs):
	print j+1, clf
	dataset_blend_test_j = np.zeros((X_sub.shape[0], len(skf)))
	for i, (train, test) in enumerate(skf):
		print "Fold", i	
		X_train = X[train]
		y_train = y[train]
		X_test = X[test]
		y_test = y[test]
		clf.fit(X_train, y_train)
		y_submission = clf.predict_proba(X_test)[:,1]
		dataset_blend_train[test, j] = y_submission
		dataset_blend_test_j[:, i] = clf.predict_proba(X_sub)[:,1]
	dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

print "Training."
clf = LogisticRegression()
clf.fit(dataset_blend_train, y)
y_submission = clf.predict_proba(dataset_blend_test)[:,1]


predict = pd.DataFrame(y_submission)
predict.to_csv('test.csv')

print "Saving Results."
with open('test.csv','rb') as i:
	with open('Output.csv','wb') as o:
		r=csv.reader(i,delimiter=',')
		w=csv.writer(o,delimiter=',')
		for f in r:
			temp=f
			if f[0]== "":
				temp[0]="Id"
				temp[1]="solved_status"
				w.writerow(temp)
				continue
			if float(f[1])>=0.80:			#The threshold 0.8 provided best results for our model
				temp[1]=1
			else:
				temp[1]=0
			w.writerow(temp)

i.close()
o.close()


