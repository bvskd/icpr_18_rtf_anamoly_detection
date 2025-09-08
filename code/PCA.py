import matplotlib.pyplot as plt
import numpy as np

from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, auc, roc_curve

data = genfromtxt("Musk.csv", delimiter = ',')
x = data[:,0:-1]
y = data[:,-1]
y[y==0] = -1

aucscore = np.zeros((20,3))

pca = PCA(n_components = 3)

for k in range (20):
    
    print('trial.%d' % k)
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.5, random_state=k)
    
    idxneg1 = np.where(Y_train == -1)[0]
    X_train_neg = X_train[idxneg1,:]
    X_train_pca = pca.fit(X_train_neg).transform(X_train)
    X_train_pca_recovered = pca.inverse_transform(X_train_pca)
    recovery_error1 = np.power(np.sum(np.power(np.subtract(X_train, X_train_pca_recovered),2), axis = 1),0.5)
    idxsort1 = np.argsort(recovery_error1)
    Y_pred1 = -1*np.ones(1531)
    Y_pred1[idxsort1[-15:-1]] = 1
    fpr1, tpr1, thresholds1 = roc_curve(Y_train, Y_pred1)
    aucscore[k,0] = auc(fpr1, tpr1)
    #cm1 = confusion_matrix(Y_train, Y_pred1)
    #dr1 = cm1[1][1]/cm1[1][1] + cm1[0][1]
    #far1 = cm1[1][0]/cm1[0][0] + cm1[1][0]
    #auc1 = auc(far1, dr1)
    
    
    X_test_pca = pca.fit(X_train_neg).transform(X_test)
    X_test_pca_recovered = pca.inverse_transform(X_test_pca)
    recovery_error2 = np.power(np.sum(np.power(np.subtract(X_test, X_test_pca_recovered),2), axis = 1),0.5)
    idxsort2 = np.argsort(recovery_error2)
    Y_pred2 = -1*np.ones(1531)
    Y_pred2[idxsort1[-15:-1]] = 1
    fpr2, tpr2, thresholds2 = roc_curve(Y_test, Y_pred2)
    aucscore[k,1] = auc(fpr2, tpr2)
    #cm2 = confusion_matrix(Y_test, Y_pred2)
    #dr2 = cm2[1][1]/cm2[1][1] + cm2[0][1]
    #far2 = cm2[1][0]/cm2[0][0] + cm2[1][0]
    #auc2 = auc(far2, dr2)
    
    idxneg2 = np.where(Y_pred2 == -1)[0]
    X_test_neg = X_test[idxneg2,:]
    X_re_train_pca = pca.fit(X_test_neg).transform(X_train)
    X_re_train_pca_recovered = pca.inverse_transform(X_re_train_pca)
    recovery_error3 = np.power(np.sum(np.power(np.subtract(X_train, X_re_train_pca_recovered),2), axis = 1),0.5)
    idxsort3 = np.argsort(recovery_error3)
    Y_pred3 = -1*np.ones(1531)
    Y_pred3[idxsort1[-15:-1]] = 1
    fpr3, tpr3, thresholds3 = roc_curve(Y_train, Y_pred3)
    aucscore[k,2] = auc(fpr3, tpr3)
    #cm3 = confusion_matrix(Y_train, Y_pred3)
    #dr3 = cm3[1][1]/cm3[1][1] + cm3[0][1]
    #far3 = cm3[1][0]/cm3[0][0] + cm3[1][0]
    #auc3 = auc(far3, dr3)

aucmean = np.mean(aucscore,axis=0)
aucvar = np.var(aucscore,axis=0)
print('\n TrainingAUC = %.4f (%f)' % (aucmean[0],aucvar[0]))
print('\n TestingAUC = %.4f (%f)' % (aucmean[1],aucvar[1]))
print('\n ReverseTrainingAUC = %.4f (%f)' % (aucmean[2],aucvar[2]))

print('\n \n Delta_tr = %.4f' % abs(aucmean[0] - aucmean[1]))
print('\n Delta_retr = %.4f' % abs(aucmean[1] - aucmean[2]))
