import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from funpack import EVALAD
#from sklearn.cluster import KMeans
#from sklearn.mixtureGaussianMixture

scaler = StandardScaler()
# please try other data sets (in the folder I sent you)
data = genfromtxt("NetwokIntrusion.csv", delimiter = ',')
x = data[:,0:-1]
y = data[:,-1]
y[y==0] = -1
idx0 = np.where(y==-1)
idx1 = np.where(y==1)

samplecount = 10000

idxarr0 = np.random.choice(idx0[0], size = int(samplecount))
idxarr1 = np.random.choice(idx1[0], size = int(samplecount))

idxsimfinal = np.concatenate((idxarr0, idxarr1), axis = 0)

x_sim = x[idxsimfinal]
y_sim = y[idxsimfinal]


# on each data set, please try the following models :
# OneClassSVM, PCA, clustering-based (k-means), denstiy-based (GMM)  
# (you may choose hyper-parameter that gives best Testing AUC)
# (but usually we should observe your metric is closer to testing error than training error)
# (for OneClassSVM, try linear first; but if your metric is not better, then try rbf kernel) 
clf = OneClassSVM(nu = 0.1, kernel='linear')

pre = np.zeros((20,3))
rec = np.zeros((20,3))
auc = np.zeros((20,3))

# this is the threshold to pick normal data 
# I think typical value is 0, but you can choose larger (so only confident normal data are used for retraining) 
thres_neg = 0 # try {0,10} and see increasing threshold may improve your metric  

for k in range (20):
    
    print('trial.%d' % k)

    X_train, X_test, Y_train, Y_test = train_test_split(x_sim, y_sim, test_size = 0.5, random_state=k)
    
    idxneg = np.where(Y_train == -1)[0]
    X_train_neg = X_train[idxneg,:]
    Y_train_neg = Y_train[idxneg]
    clf.fit(X_train_neg)
    
    adscore1 = clf.decision_function(X_train) 
    far, dr = EVALAD(Y_train,adscore1,500)
    auc[k,0] = metrics.auc(far,dr)
    
    adscore2 = clf.decision_function(X_test) 
    far, dr = EVALAD(Y_test,adscore2,500)
    auc[k,1] = metrics.auc(far,dr)
    
    adscore2 = scaler.fit(adscore2).transform(adscore2)
    idxneg = np.where(adscore2 >= thres_neg)[0]
    clf.fit(X_test[idxneg,:])
    adscore3 = clf.decision_function(X_train)
    far, dr = EVALAD(Y_train,adscore3,500)
    auc[k,2] = metrics.auc(far,dr)
    
aucmean = np.mean(auc,axis=0)
aucvar = np.var(auc,axis=0)

# please document results below in the latex tables 
# the first value is result, the value in parenthesis is variance 
# if variance is too small e.g. 0.00001, use this format: 1e-5 
print('\n TrainingAUC = %.4f (%f)' % (aucmean[0],aucvar[0]))
print('\n TestingAUC = %.4f (%f)' % (aucmean[1],aucvar[1]))
print('\n ReverseTrainingAUC = %.4f (%f)' % (aucmean[2],aucvar[2]))

print('\n \n Threshold = %f' % thres_neg)

print('\n \n Delta_tr = %.4f' % abs(aucmean[0] - aucmean[1]))
print('\n Delta_retr = %.4f' % abs(aucmean[1] - aucmean[2]))



