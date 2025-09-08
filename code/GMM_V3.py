import matplotlib.pyplot as plt
import numpy as np

from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from funpack import EVALAD

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

aucscore = np.zeros((20,3))

gm = GaussianMixture(n_components = 1, reg_covar = 1)

for k in range (20):
    
    print('trial.%d' % k)
    X_train, X_test, Y_train, Y_test = train_test_split(x_sim, y_sim, test_size = 0.5, random_state=k)
    
    idxneg1 = np.where(Y_train == -1)[0]
    X_train_neg = X_train[idxneg1,:]
    gm.fit(X_train_neg)
    loglikehood1 = -gm.score_samples(X_train) # should be inverse of anomalous score 
    far, dr = EVALAD(Y_train,loglikehood1,50) # replace 500 with 50
    aucscore[k,0] = metrics.auc(far,dr)

    
    gm.fit(X_train_neg)
    loglikehood2 = np.log(-gm.score_samples(X_test))
    far, dr = EVALAD(Y_test,loglikehood2,50) # replace 500 with 50
    aucscore[k,1] = metrics.auc(far,dr)
    idxsort2 = np.argsort(loglikehood2)
    Y_pred2 = -1*np.ones(len(Y_test))
    Y_pred2[idxsort2[-3500:-1]] = 1

    
    idxneg2 = np.where(Y_pred2 == -1)[0]
    X_test_neg = X_test[idxneg2,:]
    gm.fit(X_test_neg)
    loglikehood3 = np.log(-gm.score_samples(X_train))
    far, dr = EVALAD(Y_train,loglikehood3,50) # replace 500 with 50
    aucscore[k,2] = metrics.auc(far,dr)


aucmean = np.mean(aucscore,axis=0)
aucvar = np.var(aucscore,axis=0)
print('\n TrainingAUC = %.4f (%f)' % (aucmean[0],aucvar[0]))
print('\n TestingAUC = %.4f (%f)' % (aucmean[1],aucvar[1]))
print('\n ReverseTrainingAUC = %.4f (%f)' % (aucmean[2],aucvar[2]))

print('\n \n Delta_tr = %.4f' % abs(aucmean[0] - aucmean[1]))
print('\n Delta_retr = %.4f' % abs(aucmean[1] - aucmean[2]))