import matplotlib.pyplot as plt
import numpy as np

from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
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

pca = PCA(n_components = 3)

for k in range (20):
    
    print('trial.%d' % k)
    X_train, X_test, Y_train, Y_test = train_test_split(x_sim, y_sim, test_size = 0.5, random_state=k)
    
    idxneg1 = np.where(Y_train == -1)[0]
    X_train_neg = X_train[idxneg1,:]
    X_train_pca = pca.fit(X_train_neg).transform(X_train)
    X_train_pca_recovered = pca.inverse_transform(X_train_pca)
    recovery_error1 = np.power(np.sum(np.power(np.subtract(X_train, X_train_pca_recovered),2), axis = 1),0.5)
    far, dr = EVALAD(Y_train,recovery_error1,500)
    aucscore[k,0] = metrics.auc(far,dr)
    
    
    X_test_pca = pca.fit(X_train_neg).transform(X_test)
    X_test_pca_recovered = pca.inverse_transform(X_test_pca)
    recovery_error2 = np.power(np.sum(np.power(np.subtract(X_test, X_test_pca_recovered),2), axis = 1),0.5)
    far, dr = EVALAD(Y_test,recovery_error2,500)
    aucscore[k,1] = metrics.auc(far,dr)
    
    idxsort2 = np.argsort(recovery_error2)
    Y_pred2 = -1*np.ones(len(Y_test))
    Y_pred2[idxsort2[-4000:-1]] = 1
    
    idxneg2 = np.where(Y_pred2 == -1)[0]
    X_test_neg = X_test[idxneg2,:]
    X_re_train_pca = pca.fit(X_test_neg).transform(X_train)
    X_re_train_pca_recovered = pca.inverse_transform(X_re_train_pca)
    recovery_error3 = np.power(np.sum(np.power(np.subtract(X_train, X_re_train_pca_recovered),2), axis = 1),0.5)
    far, dr = EVALAD(Y_train,recovery_error3,500)
    aucscore[k,2] = metrics.auc(far,dr)

aucmean = np.mean(aucscore,axis=0)
aucvar = np.var(aucscore,axis=0)
print('\n TrainingAUC = %.4f (%f)' % (aucmean[0],aucvar[0]))
print('\n TestingAUC = %.4f (%f)' % (aucmean[1],aucvar[1]))
print('\n ReverseTrainingAUC = %.4f (%f)' % (aucmean[2],aucvar[2]))

print('\n \n Delta_tr = %.4f' % abs(aucmean[0] - aucmean[1]))
print('\n Delta_retr = %.4f' % abs(aucmean[1] - aucmean[2]))
