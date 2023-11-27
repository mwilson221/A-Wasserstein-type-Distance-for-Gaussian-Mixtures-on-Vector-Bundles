import numpy as np
from scipy.io import loadmat
import fdasrsf as fs
import pandas as pd
import ot

#plotting
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import seaborn as sns

#fdasrsf
from fdasrsf import fdacurve
from fdasrsf.curve_functions import innerprod_q2

def load_KMKM_data(t,folder_name="",file_name="KMKM_results.mat"):
    X = loadmat(folder_name+file_name)['X_list'][0][t]
    X = X.reshape((2,X.shape[0]//2,X.shape[1]))
    mode_idx= (loadmat(folder_name+file_name)['mode_idx_list'][0][t].astype(int)-1)[0]
    idx = (loadmat(folder_name+file_name)['idx_list'][0][t].astype(int)-1)[0]

    return (X,idx,mode_idx)


def init_fourier_basis(n,N):
    t=np.arange(0,2*np.pi,(2*np.pi)/(N))

    B=np.zeros((n,N,(2*(N-1))))

    B[0,:,0] = np.ones(N)
    B[1,:,1] = np.ones(N)

    for k in range(1,N//2):
        B[0,:,4*k-2] = np.sqrt(2)*(np.sin(k*t))
        B[1,:,4*k-1] = np.sqrt(2)*(np.sin(k*t)) 
        B[0,:,4*k]   = np.sqrt(2)*(np.cos(k*t))
        B[1,:,4*k+1] = np.sqrt(2)*(np.cos(k*t)) 
                 
    return B


def align_by_cluster(beta,idx,N=200, scale = False, parallel=False):
    
    if beta.shape[2] != len(idx):
        raise ValueError("Number of curves ("+str(beta.shape[2])+") does not match length of index ("+str(len(idx))+").")
    
    F=[]

    for i in range(max(idx)+1):
        F.append(fdacurve(beta[:,:,idx==i], mode='C', N=N, scale=scale))
        F[i].srvf_align(parallel=True)
        
    return F


def function_project(B,functions):
    if len(functions.shape)==3:
        return np.array([[innerprod_q2(B[:,:,i],functions[:,:,j]) for j in range(functions.shape[2])] for i in range(B.shape[2])] )
    else:
        return np.array([innerprod_q2(B[:,:,i],functions) for i in range(B.shape[2])] )
        

class WrappedGaussianMixture():
    
    def __init__(self, manifold, p_ref=None):
        self.manifold = manifold
        self.dim = manifold.dim
        self.p_ref = p_ref
        
        self.init_moving_frame()
    
    
    def init_moving_frame(self):
        if self.p_ref==None:
            E = np.eye(self.dim-2)
            self.p_ref = E[:,0].T
            self.F_ref = E[:,1:].T
        else:
            print("Not working")
            # E = np.linalg.qr(np.concatenate([self.p_ref,np.eye(self.dim)[:,1:]]))[0]
            # self.p_ref = E[:,0]
            # self.F_ref = E[:,1:]
    
    
    def fit_frame(self, means, data):
        
        self.n_components=len(means)
        
        weights=[]
        covariances=[]
        
        for i in range(self.n_components):
            F_m = self.manifold.metric.parallel_transport(self.F_ref,self.p_ref,end_point=means[i])
            V = self.manifold.metric.log(data[i].T, means[i])
            # print("F_m shape: "+str(F_m.shape))
            # print("V_shape: " +str(V.shape))
            covariances.append(np.cov(F_m@V.T))
            
        self.weights = np.array([data[i].shape[1]/sum([d.shape[1] for d in data]) for i in range(self.n_components)])
        self.means = np.array(means)
        self.covariances = np.array(covariances)
            


#Wasserstein-type distance
def wasserstein_type_distance(mu0,mu1):
    d_M=np.zeros((mu0.n_components,mu1.n_components))
    d_B=np.zeros((mu0.n_components,mu1.n_components))

    for i in range(mu0.n_components):
        (m0,sigma0)= (mu0.means[i],mu0.covariances[i])
        for j in range(mu1.n_components):
            (m1,sigma1)= (mu1.means[j],mu1.covariances[j])

            # #normalize just to make sure - generally ~0.99999
            # print(np.linalg.norm(m0))
            # print(np.linalg.norm(m1))
            m0=m0/np.linalg.norm(m0)
            m1=m1/np.linalg.norm(m1)

            d_M[i,j] = np.arccos(m0@m1)

            # #use svd to calculate square roots - no issues with complex values
            u0,s0,v0 = np.linalg.svd(sigma0)
            sqrt_sigma0 = u0@np.diag(np.sqrt(s0))@u0.T
            cross_term = sqrt_sigma0 @ sigma1 @ sqrt_sigma0
            u2,s2,v2 = np.linalg.svd(cross_term)
            sqrt_cross_term = u2@np.diag(np.sqrt(s2))@v2

            d_B[i,j] = np.trace(sigma0 + sigma1 -2*sqrt_cross_term)
            # d_B[i,j] = np.real(np.trace(sigma0 + sigma1 -2*scipy.linalg.sqrtm(scipy.linalg.sqrtm(sigma0) + sigma1 + scipy.linalg.sqrtm(sigma0))))
      
    M = d_M**2 + d_B 
    
    return ot.emd2(mu0.weights,mu1.weights,M)#, ot.emd2(mu0.weights,mu1.weights,d_M**2), ot.emd2(mu0.weights,mu1.weights,d_B)



#E-Divisive - Energy Distance Change point analysis
#Author: https://github.com/egoolish/ecp/
from scipy.spatial.distance import pdist, squareform
import copy

def e_divisive(D, sig_lvl = 0.05, R = 199, k = None, min_size = 30, alpha = 1):
    """Test: documentation goes here."""

    #Checks for invalid arguments.
    if (R < 0) and (k is None):
        raise ValueError("R must be a nonnegative integer.")
    if (sig_lvl <= 0 or sig_lvl >= 1) and (k is None):
        raise ValueError("sig_lvl must be a positive real number between 0 and 1.")
    if (min_size < 2):
        raise ValueError("min_size must be an integer greater than 1.")
    if (alpha > 2 or alpha <= 0):
        raise ValueError("alpha must be in (0, 2].")
    
    #Length of time series
    n = (D.shape)[0]

    if (k is None):
        k = n
    else:
        #No need to perform the permutation test
        R = 0
    
    ret = {"k_hat": 1}
    pvals = []
    permutations = []
    changes = [0, n] #List of change-points
    energy = np.zeros((n, 2)) #Matrix used to avoid recalculating statistics
    # D = np.power(squareform(pdist(X)), alpha)
    D = np.power(D, alpha)
    con = -1

    while (k > 0):
        tmp_data = e_split(changes, D, min_size, False, energy) #find location for change point
        
        e_stat = tmp_data["best"]
        tmp = copy.deepcopy(tmp_data["changes"])
        #newest change-point
        con = tmp[-1]

        #Not able to meet minimum size constraint
        if(con == -1):
            break 
        #run permutation test
        result = sig_test(D, R, changes, min_size, e_stat)

        pval = result[0] #approximate p-value
        permutations.append(result[1]) #number of permutations performed
        pvals.append(pval) #list of computed p-values
        
        #change point not significant
        if(pval > sig_lvl):
            break
        
        changes = copy.deepcopy(tmp) #update set of change points
        ret["k_hat"] = ret["k_hat"]+1 #update number of clusters
        k = k-1

    estimates = copy.deepcopy(changes)
    estimates.sort()
    ret["order_found"] = changes
    ret["estimates"] = estimates
    ret["considered_last"] = con
    ret["p_values"] = pvals
    ret["permutations"] = permutations
    ret["cluster"] = np.repeat(np.arange(0,len(np.diff(estimates))), np.diff(estimates))
    return ret

def e_split(changes, D, min_size, for_sim = False, energy = None):
    changes = copy.deepcopy(changes)
    splits = copy.deepcopy(changes)
    splits.sort()
    best = [-1, float('-inf')]
    ii = -1
    jj = -1

    #If procedure is being used for significance test
    if(for_sim):

        #Iterate over intervals
        for i in range(1, len(splits)):
            tmp = splitPoint(splits[i-1], splits[i] - 1, D, min_size)
            #tmp[1] is the "energy released" when the cluster was split
            if(tmp[1]>best[1]):
                ii = splits[i-1]
                jj = splits[i] - 1
                best = tmp #update best split point found so far
        #update the list of changepoints
        changes.append(int(best[0]))
        return {"start": ii, "end": jj, "changes": changes, "best": best[1]}
    else:
        if(energy is None):
            raise ValueError("Must specify one of: for_sim, energy")
        
        #iterate over intervals
        for i in range(1, len(splits)):
            if(energy[splits[i-1],0]):
                tmp = energy[splits[i-1],:]
            else:
                tmp = splitPoint(splits[i-1], splits[i]-1, D, min_size)
                energy[splits[i-1], 0] = tmp[0]
                energy[splits[i-1], 1] = tmp[1]

            #tmp[1] is the "energy released" when the cluster was split
            if(tmp[1] > best[1]):
                ii = splits[i-1]
                jj = splits[i] - 1
                best = tmp
            
        changes.append(int(best[0])) #update the list of change points
        energy[ii, 0] = 0 #update matrix to account for newly proposed change point
        energy[ii, 1] = 0 #update matrix to account for newly proposed change point
        return {"start": ii, "end": jj, "changes": changes, "best": best[1]}

def splitPoint(start, end, D, min_size):
    #interval too small to split
    if(end-start+1 < 2*min_size):
        return [-1, float("-inf")]
    
    #use needed part of distance matrix
    D = D[start:end+1, start:end+1]
    return splitPointpseudoC(start, end, D, min_size)
    
def splitPointpseudoC(s, e, D, min_size):
    """ This function used to be written in C++. However, it used SEXP to return
        a numeric vector data type, which is incompatible with Python. As such, 
        the function is temporarily rewritten in Python, but could be made faster
        by replacing this with a Python to C++ call.
    """
    best = [-1.0, float("-inf")]
    e = e - s + 1
    t1 = min_size
    t2 = min_size <<1
    cut1 = D[0:t1, 0:t1]
    cut2 = D[t1:t2, t1:t2]
    cut3 = D[0:t1, t1:t2]
    A = np.sum(cut1)/2
    B1 = np.sum(cut2)/2
    AB1 = np.sum(cut3)
    tmp = 2*AB1/((t2-t1)*(t1)) - 2*B1/((t2-t1-1)*(t2-t1)) - 2*A/((t1-1)*(t1))
    tmp *= (t1*(t2-t1)/t2)
    if(tmp > best[1]):
        best[0] = t1 + s
        best[1] = tmp
    t2+=1
    B = np.full(e+1, B1)
    AB = np.full(e+1, AB1)
    while(t2 <= e):
        B[t2] = B[t2 - 1] + np.sum(D[t2-1, t1:t2-1])
        AB[t2] = AB[t2-1] + np.sum(D[t2-1, 0:t1])
        tmp = 2*AB[t2]/((t2-t1)*(t1))-2*B[t2]/((t2-t1-1)*(t2-t1))-2*A/((t1)*(t1-1))
        tmp *= (t1*(t2-t1)/t2)
        if (tmp > best[1]):
            best[0] = t1 + s
            best[1] = tmp
        t2 += 1
    
    t1 += 1

    while(True):
        t2 = t1+ min_size
        if(t2 > e):
            break
        addA = np.sum(D[t1-1, 0:t1-1])
        A += addA
        addB = np.sum(D[t1-1, t1:t2-1])
        while(t2 <= e):
            addB += D[t1-1, t2-1]
            B[t2] -= addB
            AB[t2] += (addB-addA)
            tmp = 2*AB[t2]/((t2-t1)*(t1))-2*B[t2]/((t2-t1-1)*(t2-t1))-2*A/((t1-1)*(t1))
            tmp *= (t1*(t2-t1)/t2)
            if(tmp > best[1]):
                best[0] = t1+s
                best[1] = tmp
            t2 += 1
        t1 += 1
    return best     


def sig_test(D, R, changes, min_size, obs):
    #No permutations, so return a p-value of 0
    if(R == 0): 
        return [0, 0]
    over = 0
    for _ in range(R):
        Dcopy = np.copy(D)
        changes_copy = copy.deepcopy(changes)
        changes_copy2 = copy.deepcopy(changes)
        D1 = perm_cluster(Dcopy, changes_copy) #permute within cluster
        tmp = e_split(changes_copy2, D1, min_size, True)
        if(tmp['best'] >= obs):
            over = over + 1
    pval = (1 + over)/ float((R + 1))
    return [pval, R]

def perm_cluster(D, points):
    points.sort()
    K = len(points) - 1 #number of clusters
    for i in range(K):
        u = np.arange(points[i], points[i+1])
        np.random.shuffle(u)
        Dtmp = D[points[i]:points[i+1], points[i]:points[i+1]]
        Dtmp[0:points[i+1]-points[i], :] = Dtmp[u-points[i], :]
        Dtmp[:, 0:points[i+1]-points[i]] = Dtmp[:, u-points[i]]
        D[points[i]:points[i+1], points[i]:points[i+1]] = Dtmp
    return D



# Plotting
def plot_frame(X,mode_idx,idx):
    c = sns.color_palette()[:len(mode_idx)]+['black']

    plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    for i in range(X.shape[2]):
        plt.plot(X[0,:,i],X[1,:,i],c = 'black')
    plt.xlim((0,500))
    plt.ylim((0,500))
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1,2,2)
    for i in range(X.shape[2]):
        if i in mode_idx:
            plt.plot(X[0,:,i],X[1,:,i],c=c[idx[i]],linewidth=4)
        else:
            plt.plot(X[0,:,i],X[1,:,i],c=c[idx[i]])
    plt.xlim((0,500))
    plt.ylim((0,500))
    plt.xticks([])
    plt.yticks([])
    
def mds_plot(MW, cp_hat):
    mds = MDS(n_components=2, dissimilarity='precomputed', normalized_stress='auto')

    embedding2 = mds.fit_transform(MW)

    plt.plot(embedding2[:,0],embedding2[:,1],'black',linewidth=.5,alpha=.35)

    for t in range(62):
        if t<cp_hat:
            plt.plot(embedding2[t,0],embedding2[t,1],'.',c='red')
        elif t==cp_hat:
            plt.plot(embedding2[t,0],embedding2[t,1],'*',c='black',linewidth=4)
        else:
            plt.plot(embedding2[t,0],embedding2[t,1],'.',c='blue')

    plt.title("MDS plot")
    plt.xlim(-3.5,3.5)
    plt.ylim(-3.5,3.5)