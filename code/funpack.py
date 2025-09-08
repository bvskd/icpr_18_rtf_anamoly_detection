
import numpy as np

from scipy.linalg import eigh, inv

from sklearn.metrics.pairwise import pairwise_kernels,paired_cosine_distances


# -------------------------------------------------
# linear co-regularized amomaly detection technique
# -------------------------------------------------
def COREGAD(L1,L2,Yl,U1,U2,Z1,Z2,T1,T2,lam1,lam2,lam3):

    A1 = np.dot(np.transpose(L1),L1) + lam1*np.identity(L1.shape[1]) + lam2*np.dot(np.transpose(U1),U1) - lam3*np.dot(np.transpose(Z1),Z1)
    A2 = np.dot(np.transpose(L2),L2) + lam1*np.identity(L2.shape[1]) + lam2*np.dot(np.transpose(U2),U2) - lam3*np.dot(np.transpose(Z2),Z2)
    B1 = np.transpose(L1).dot(Yl)
    B2 = np.transpose(L2).dot(Yl)
    C1 = lam2*np.dot(np.transpose(U2),U1) - lam3*np.dot(np.transpose(Z2),Z1)
    C2 = lam2*np.dot(np.transpose(U1),U2) - lam3*np.dot(np.transpose(Z1),Z2)
    
    beta1 = np.dot(inv(A1 - (C2.dot(inv(A2))).dot(C1)), B1+(C2.dot(inv(A2))).dot(B2))
    beta2 = np.dot(inv(A2 - (C1.dot(inv(A1))).dot(C2)), B2+(C1.dot(inv(A1))).dot(B1))
    
    label_pred1 = T1.dot(beta1)
    label_pred2 = T2.dot(beta2)
    
    adscore = abs(label_pred1 - label_pred2)            
    
    return adscore 

# -------------------------------------------------
# kernel co-regularized amomaly detection technique
# -------------------------------------------------
def COREGAD_Kernel(L1,L2,Yl,U1,U2,Z1,Z2,T1,T2,lam1,lam2,lam3,metric_kernel,gam1,gam2):
    
    s1 = np.r_[L1,U1,Z1]
    s2 = np.r_[L2,U2,Z2]
    
    K1x = pairwise_kernels(L1,s1,metric=metric_kernel,gamma = gam1)
    K2x = pairwise_kernels(L2,s2,metric=metric_kernel,gamma = gam2)
    K1u = pairwise_kernels(U1,s1,metric=metric_kernel,gamma = gam1)
    K2u = pairwise_kernels(U2,s2,metric=metric_kernel,gamma = gam2)
    K1z = pairwise_kernels(Z1,s1,metric=metric_kernel,gamma = gam1)
    K2z = pairwise_kernels(Z2,s2,metric=metric_kernel,gamma = gam2)
    G1 = pairwise_kernels(s1,metric=metric_kernel,gamma = gam1)
    G2 = pairwise_kernels(s2,metric=metric_kernel,gamma = gam2)
    K1t = pairwise_kernels(T1,s1,metric=metric_kernel,gamma = gam1)
    K2t = pairwise_kernels(T2,s2,metric=metric_kernel,gamma = gam2)
    
    A1 = np.dot(np.transpose(K1x),K1x) + lam1*G1 + 1e1*np.identity(G1.shape[0]) + lam2*np.dot(np.transpose(K1u),K1u) - lam3*np.dot(np.transpose(K1u),K1u)
    A2 = np.dot(np.transpose(K2x),K2x) + lam1*G2 + 1e1*np.identity(G2.shape[0]) + lam2*np.dot(np.transpose(K2u),K2u) - lam3*np.dot(np.transpose(K2u),K2u)
    B1 = np.transpose(K1x).dot(Yl)
    B2 = np.transpose(K2x).dot(Yl)
    C1 = lam2*np.dot(np.transpose(K2u),K1u) - lam3*np.dot(np.transpose(K2z),K1z)
    C2 = lam2*np.dot(np.transpose(K1u),K2u) - lam3*np.dot(np.transpose(K1z),K2z)
    
    beta1 = np.dot(inv(A1 - (C2.dot(inv(A2))).dot(C1)), B1+(C2.dot(inv(A2))).dot(B2))
    beta2 = np.dot(inv(A2 - (C1.dot(inv(A1))).dot(C2)), B2+(C1.dot(inv(A1))).dot(B1))
    
    label_pred1 = K1t.dot(beta1)
    label_pred2 = K2t.dot(beta2)
    
    adscore = abs(label_pred1 - label_pred2)            
            
    return adscore 
            

# -------------------------------------------------
# HOAD [Gao, ICDM 2011]
# -------------------------------------------------
def HOAD(SimV1,SimV2,k,keval,m):
    
    n = SimV1.shape[0]

    Z = np.block([
                [SimV1, m*np.identity(n)],
                [m*np.identity(n), SimV2]
                ])
    
    D = np.diag(np.sum(Z,axis=1))
    
    L = D - Z 
    
    w,vr = eigh(L,eigvals=(0,k-1))
    
    adscore = np.zeros((n,len(keval)))
    
    for i in range(len(keval)):
    
        Hv1 = vr[0:n,0:keval[i]]
        Hv2 = vr[n:,0:keval[i]]
    
        adscore[:,i] = 1 - paired_cosine_distances(Hv1,Hv2)
    
    return adscore 

# -----------------------------------------
# evaluate ROC of anomaly detection results 
# -----------------------------------------
def EVALAD(label_true,adscore,numbins):
    
    num_test = len(label_true)
    
    thres = np.linspace(0,num_test,numbins)
    thres = thres.astype(int)
    thres = np.delete(thres,-1)
    thres = np.delete(thres,-1)
    
    idx_adsort = np.argsort(adscore,axis=0)
    
#    
    detectionrate = np.zeros(len(thres))
    falsealarmrate = np.zeros(len(thres)) 
    
    for i in range(len(thres)):
        
        label_pred = np.zeros(num_test)
        
        label_pred[idx_adsort[0:thres[i]]] = -1
        
        label_pred[label_pred!=-1] = 1
        
        fpr = np.sum(np.logical_and(label_pred==1,label_true==-1)) 
        fnr = np.sum(np.logical_and(label_pred==-1,label_true==1)) 
        tpr = np.sum(np.logical_and(label_pred==1,label_true==1)) 
        tnr = np.sum(np.logical_and(label_pred==-1,label_true==-1)) 
        
        falsealarmrate[i] = fpr / (fpr+tnr) 
        detectionrate[i] = tpr / (tpr+fnr) 
        
    return falsealarmrate,detectionrate


# -----------------------------------------
# evaluate ROC of anomaly detection results 
# -----------------------------------------
def EVALFair(label_true, label_pred, idxg):
    
    idxg0 = np.where(idxg==0)[0]
    idxg1 = np.where(idxg==1)[0]
    
    idxp_pred = np.where(label_pred==1)[0]
    
    idxp_true = np.where(label_true==1)[0]
    idxn_true = np.where(label_true==-1)[0]
    
    idxp0_true = np.intersect1d(idxg0,idxp_true)
    idxp1_true = np.intersect1d(idxg1,idxp_true)
    
    idxp0_true_pospred = np.intersect1d(idxp0_true,idxp_pred)
    idxp1_true_pospred = np.intersect1d(idxp1_true,idxp_pred)
    
    prob_p0 = len(idxp0_true_pospred) / len(idxp0_true)
    prob_p1 = len(idxp1_true_pospred) / len(idxp1_true)
    
    disparity_pos = prob_p0 / prob_p1
    
    idxn0_true = np.intersect1d(idxg0,idxn_true)
    idxn1_true = np.intersect1d(idxg1,idxn_true)
    
    idxp0_true_negpred = np.intersect1d(idxn0_true,idxp_pred)
    idxp1_true_negpred = np.intersect1d(idxn1_true,idxp_pred)
    
    prob_n0 = len(idxp0_true_negpred) / len(idxn0_true)
    prob_n1 = len(idxp1_true_negpred) / len(idxn1_true)
    
    disparity_neg = prob_n0 / prob_n1
    
    return disparity_pos,disparity_neg