import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score
from funpack import EVALFair

# please try other data sets (in the folder I sent you)
data = genfromtxt("NetwokIntrusion.csv", delimiter = ',')
x = data[:,0:-1]
y = data[:,-1]
y[y==0] = -1
idx0 = np.where(y==-1)
idx1 = np.where(y==1)
idx2 = np.where(y==2)
idx3 = np.where(y==3)
idx4 = np.where(y==4)

samplecount = 2000

idxarr0 = np.random.choice(idx0[0], size = int(samplecount))
idxarr1 = np.random.choice(idx1[0], size = int(samplecount))
idxarr2 = np.random.choice(idx2[0], size = int(samplecount))
idxarr3 = np.random.choice(idx3[0], size = int(samplecount))
idxarr4 = np.random.choice(idx4[0], size = int(samplecount))

idxsimfinal = np.concatenate((idxarr0, idxarr1, idxarr2, idxarr3, idxarr4), axis = 0)

x_sim = x[idxsimfinal]
y_sim = y[idxsimfinal]

#print('Logistic Regression')
##on each data set, please try the following models 
##logistic regression, linear SVM, SVC (default kernel), k-NN, decision tree, random forest 
##(you may choose hyper-parameter that gives best Testing Error)
##(but usually we should observe your metric is closer to testing error than training error)
##clf = SVC(C = 1, kernel='linear') # {1,1e1}
#clf = LogisticRegression(C = 4444) # {1,1e1}
#
#err = np.zeros((20,3))
#f1 = np.zeros((20,3))
#dsp_pos = np.zeros((20,3))
#dsp_neg = np.zeros((20,3))
#
#for k in range (20):
#    
#    print('trial.%d' % k)
#    
#    X_train, X_test, Y_train, Y_test = train_test_split(x_sim, y_sim, test_size = 0.5, random_state=k)
#    clf.fit(X_train, Y_train)
#    
#    Y_train_pred = clf.predict(X_train) 
#    err[k,0] = 1- accuracy_score(Y_train,Y_train_pred)
#    f1[k,0] = f1_score(Y_train, Y_train_pred, average='weighted') 
#    
#    Y_pred = clf.predict(X_test) 
#    err[k,1] = 1- accuracy_score(Y_test,Y_pred)
#    f1[k,1] = f1_score(Y_test, Y_pred, average='weighted') 
#    
#    Y_conf = abs(clf.decision_function(X_test))
#    idx_conf = np.argsort(Y_conf)[::-1]
#    
#    clf.fit(X_test, Y_pred)
#    Y_train_pred = clf.predict(X_train) 
#    err[k,2] = 1-accuracy_score(Y_train,Y_train_pred)
#    f1[k,2] = f1_score(Y_train, Y_train_pred, average='weighted') 
#    
#errmean = np.mean(err,axis=0)
#errvar = np.var(err,axis=0)
#
#f1mean = np.mean(f1,axis=0)
#f1var = np.var(f1,axis=0)
#
## please document results below in the latex tables 
## the first value is result, the value in parenthesis is variance 
## if variance is too small e.g. 0.00001, use this format: 1e-5 
#print('\n TrainingErr = %.4f (%f)' % (errmean[0],errvar[0]))
#print('\n TestingErr = %.4f (%f)' % (errmean[1],errvar[1]))
#print('\n ReverseTrainingErr = %.4f (%f)' % (errmean[2],errvar[2]))
#
#print('\n \n TrainingF1 = %.4f (%f)' % (f1mean[0],f1var[0]))
#print('\n TestingF1 = %.4f (%f)' % (f1mean[1],f1var[1]))
#print('\n ReverseTrainingF1 = %.4f (%f)' % (f1mean[2],f1var[2]))
#
#print('\n \n Delta_tr(Err) = %.4f' % abs(errmean[0] - errmean[1]))
#print('\n Delta_retr(Err) = %.4f' % abs(errmean[1] - errmean[2]))
#
#print('\n \n Delta_tr(F1) = %.4f' % abs(f1mean[0] - f1mean[1]))
#print('\n Delta_retr(F1) = %.4f' % abs(f1mean[1] - f1mean[2]))

print('LSVM')
clf = SVC(kernel='linear', C = 1) # {1,1e1}

err = np.zeros((20,3))
f1 = np.zeros((20,3))
dsp_pos = np.zeros((20,3))
dsp_neg = np.zeros((20,3))

for k in range (20):
    
    print('trial.%d' % k)

    X_train, X_test, Y_train, Y_test = train_test_split(x_sim, y_sim, test_size = 0.5, random_state=k)
    clf.fit(X_train, Y_train)
    
    Y_train_pred = clf.predict(X_train) 
    err[k,0] = 1- accuracy_score(Y_train,Y_train_pred)
    f1[k,0] = f1_score(Y_train, Y_train_pred, average='weighted') 
    
    Y_pred = clf.predict(X_test) 
    err[k,1] = 1- accuracy_score(Y_test,Y_pred)
    f1[k,1] = f1_score(Y_test, Y_pred, average='weighted') 

    Y_conf = abs(clf.decision_function(X_test))
    idx_conf = np.argsort(Y_conf)[::-1]
    
    clf.fit(X_test, Y_pred)
    Y_train_pred = clf.predict(X_train) 
    err[k,2] = 1-accuracy_score(Y_train,Y_train_pred)
    f1[k,2] = f1_score(Y_train, Y_train_pred, average='weighted') 
    
errmean = np.mean(err,axis=0)
errvar = np.var(err,axis=0)

f1mean = np.mean(f1,axis=0)
f1var = np.var(f1,axis=0)

# please document results below in the latex tables 
# the first value is result, the value in parenthesis is variance 
# if variance is too small e.g. 0.00001, use this format: 1e-5 
print('\n TrainingErr = %.4f (%f)' % (errmean[0],errvar[0]))
print('\n TestingErr = %.4f (%f)' % (errmean[1],errvar[1]))
print('\n ReverseTrainingErr = %.4f (%f)' % (errmean[2],errvar[2]))

print('\n \n TrainingF1 = %.4f (%f)' % (f1mean[0],f1var[0]))
print('\n TestingF1 = %.4f (%f)' % (f1mean[1],f1var[1]))
print('\n ReverseTrainingF1 = %.4f (%f)' % (f1mean[2],f1var[2]))

print('\n \n Delta_tr(Err) = %.4f' % abs(errmean[0] - errmean[1]))
print('\n Delta_retr(Err) = %.4f' % abs(errmean[1] - errmean[2]))

print('\n \n Delta_tr(F1) = %.4f' % abs(f1mean[0] - f1mean[1]))
print('\n Delta_retr(F1) = %.4f' % abs(f1mean[1] - f1mean[2]))

#print('KNN')
#clf = KNeighborsClassifier(n_neighbors = 3) # {1,1e1}
#
#err = np.zeros((20,3))
#f1 = np.zeros((20,3))
#dsp_pos = np.zeros((20,3))
#dsp_neg = np.zeros((20,3))
#
#for k in range (20):
#    
#    print('trial.%d' % k)
#
#    X_train, X_test, Y_train, Y_test = train_test_split(x_sim, y_sim, test_size = 0.5, random_state=k)
#    clf.fit(X_train, Y_train)
#    
#    Y_train_pred = clf.predict(X_train) 
#    err[k,0] = 1- accuracy_score(Y_train,Y_train_pred)
#    f1[k,0] = f1_score(Y_train, Y_train_pred, average='weighted') 
#    
#    Y_pred = clf.predict(X_test) 
#    err[k,1] = 1- accuracy_score(Y_test,Y_pred)
#    f1[k,1] = f1_score(Y_test, Y_pred, average='weighted') 
#
#    Y_conf = abs(clf.predict_proba(X_test))
#    idx_conf = np.argsort(Y_conf)[::-1]
#    
#    clf.fit(X_test, Y_pred)
#    Y_train_pred = clf.predict(X_train) 
#    err[k,2] = 1-accuracy_score(Y_train,Y_train_pred)
#    f1[k,2] = f1_score(Y_train, Y_train_pred, average='weighted') 
#    
#errmean = np.mean(err,axis=0)
#errvar = np.var(err,axis=0)
#
#f1mean = np.mean(f1,axis=0)
#f1var = np.var(f1,axis=0)
#
## please document results below in the latex tables 
## the first value is result, the value in parenthesis is variance 
## if variance is too small e.g. 0.00001, use this format: 1e-5 
#print('\n TrainingErr = %.4f (%f)' % (errmean[0],errvar[0]))
#print('\n TestingErr = %.4f (%f)' % (errmean[1],errvar[1]))
#print('\n ReverseTrainingErr = %.4f (%f)' % (errmean[2],errvar[2]))
#
#print('\n \n TrainingF1 = %.4f (%f)' % (f1mean[0],f1var[0]))
#print('\n TestingF1 = %.4f (%f)' % (f1mean[1],f1var[1]))
#print('\n ReverseTrainingF1 = %.4f (%f)' % (f1mean[2],f1var[2]))
#
#print('\n \n Delta_tr(Err) = %.4f' % abs(errmean[0] - errmean[1]))
#print('\n Delta_retr(Err) = %.4f' % abs(errmean[1] - errmean[2]))
#
#print('\n \n Delta_tr(F1) = %.4f' % abs(f1mean[0] - f1mean[1]))
#print('\n Delta_retr(F1) = %.4f' % abs(f1mean[1] - f1mean[2]))
#
#print('Decision Tree')
#clf = DecisionTreeClassifier(max_depth = 7) # {1,1e1}
#
#err = np.zeros((20,3))
#f1 = np.zeros((20,3))
#dsp_pos = np.zeros((20,3))
#dsp_neg = np.zeros((20,3))
#
#for k in range (20):
#    
#    print('trial.%d' % k)
#
#    X_train, X_test, Y_train, Y_test = train_test_split(x_sim, y_sim, test_size = 0.5, random_state=k)
#    clf.fit(X_train, Y_train)
#    
#    Y_train_pred = clf.predict(X_train) 
#    err[k,0] = 1- accuracy_score(Y_train,Y_train_pred)
#    f1[k,0] = f1_score(Y_train, Y_train_pred, average='weighted') 
#    
#    Y_pred = clf.predict(X_test) 
#    err[k,1] = 1- accuracy_score(Y_test,Y_pred)
#    f1[k,1] = f1_score(Y_test, Y_pred, average='weighted') 
#
#    Y_conf = abs(clf.predict(X_test))
#    idx_conf = np.argsort(Y_conf)[::-1]
#    
#    clf.fit(X_test, Y_pred)
#    Y_train_pred = clf.predict(X_train) 
#    err[k,2] = 1-accuracy_score(Y_train,Y_train_pred)
#    f1[k,2] = f1_score(Y_train, Y_train_pred, average='weighted') 
#    
#errmean = np.mean(err,axis=0)
#errvar = np.var(err,axis=0)
#
#f1mean = np.mean(f1,axis=0)
#f1var = np.var(f1,axis=0)
#
## please document results below in the latex tables 
## the first value is result, the value in parenthesis is variance 
## if variance is too small e.g. 0.00001, use this format: 1e-5 
#print('\n TrainingErr = %.4f (%f)' % (errmean[0],errvar[0]))
#print('\n TestingErr = %.4f (%f)' % (errmean[1],errvar[1]))
#print('\n ReverseTrainingErr = %.4f (%f)' % (errmean[2],errvar[2]))
#
#print('\n \n TrainingF1 = %.4f (%f)' % (f1mean[0],f1var[0]))
#print('\n TestingF1 = %.4f (%f)' % (f1mean[1],f1var[1]))
#print('\n ReverseTrainingF1 = %.4f (%f)' % (f1mean[2],f1var[2]))
#
#print('\n \n Delta_tr(Err) = %.4f' % abs(errmean[0] - errmean[1]))
#print('\n Delta_retr(Err) = %.4f' % abs(errmean[1] - errmean[2]))
#
#print('\n \n Delta_tr(F1) = %.4f' % abs(f1mean[0] - f1mean[1]))
#print('\n Delta_retr(F1) = %.4f' % abs(f1mean[1] - f1mean[2]))