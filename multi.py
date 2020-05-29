from numpy import linalg as LA
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
import commFunc
import random

def genBer(A,n_k,symm=True):
    k = len(n_k)
    res = np.zeros((1,sum(n_k)))
    for i in np.arange(k):
        block = A[i,0]*np.ones((n_k[i],n_k[0]))
        for j in np.arange(1,k):
            block = np.concatenate((block, A[i,j]*np.ones((n_k[i],n_k[j]))), axis=1)
        res = np.concatenate((res,block),axis=0)
    res = res[1:,:]
    
    if symm:
        gen = np.tril(res,-1)
        gen = np.random.binomial(1,gen)
        gen = gen+gen.T
    else:
        gen = np.random.binomial(1,res)
    return res,gen

class SCME:
    
    def __init__(self,Alist,k,method='gmm'):
        self.A = Alist
        self.k = k
        self.L = len(Alist)
        self.method = method
    
    def singleUpdate(self,lamb,lr0=1,th=1e-4,iteration=30,decay=1,epsilon=1e-3,n_init=10):
        
        lamb = [float(i) for i in lamb]
        
        Amean = sum(self.A)/(self.L-1)
        t = 0
        
        while t<iteration:
            
            lr = lr0/(1+decay*t)
            t += 1
            
            ## Calculate eigenvalue
            
            evals, evecs = largest_eigsh(sum([a*b for a,b in zip(lamb,self.A)]), self.k+1, which='LM')
            evals_abs = abs(evals)
            evecs = evecs[:,evals_abs.argsort()]
            evals = evals[evals_abs.argsort()]
            
            ## Calculate gradient
            
            grad_k = np.array([evecs[:,1].dot(x-Amean).dot(evecs[:,1])*np.sign(evals[1]) for x in self.A])*self.L/(self.L-1)
            grad_k1 = np.array([evecs[:,0].dot(x-Amean).dot(evecs[:,0])*np.sign(evals[0]) for x in self.A])*self.L/(self.L-1)
            grad = (grad_k*abs(evals[0])-grad_k1*abs(evals[1]))/evals[0]**2
            
            #print(t,':',lamb,grad)
            ## Update lambda if gradient large enough, o.w. break
            
            if sum(grad**2)**0.5<th:
                break
            else:
                grad /= sum(grad**2)**0.5
            
            lamb = lamb+lr*grad;
            lamb = abs(lamb)/sum(lamb)
        
        eratio = abs(evals[1])/abs(evals[0])
        
        if self.method == 'gmm':
            label = GaussianMixture(n_components=self.k,n_init=n_init).fit_predict(np.real(evecs[:,1:]))
        else :
            label = KMeans(n_clusters=self.k,n_init=n_init).fit_predict(np.real(evecs[:,1:]))
        
        return(label,lamb,eratio)
    
    def multipleUpdate(self,itr=5,lr0=0.5,th=1e-4,iteration=30,decay=1,epsilon=1e-3,n_init=10):
        if self.L==1:
            ## Calculate eigenvalue
            evals, evecs = largest_eigsh(self.A[0].astype(float), self.k+1, which='LM')
            evals_abs = abs(evals)
            evecs = evecs[:,evals_abs.argsort()]
            evals = evals[evals_abs.argsort()]
            if self.method == 'gmm':
                label = GaussianMixture(n_components=self.k,n_init=n_init).fit_predict(np.real(evecs[:,1:]))
            else:
                label = KMeans(n_clusters=self.k,n_init=n_init).fit_predict(np.real(evecs[:,1:]))
            
            return(label,[1],evals_abs[1]/evals_abs[0])
        
        eratio_res = 0
        
        for i in np.arange(itr):
            
            ## Random Initialization
            lamb = np.random.randint(low=1,high=10,size=self.L)
            lamb = abs(lamb)/sum(lamb)
            
            label,lamb,eratio = self.singleUpdate(lamb,lr0,th,iteration,decay,epsilon,n_init)
            if eratio>eratio_res:
                eratio_res,label_res,lamb_res = eratio,label,lamb
        
        return(label_res,lamb_res,eratio_res)

class ISC:
    
    def __init__(self,Alist,k,method='gmm'):
        self.Alist = Alist
        self.k = k
        self.n = Alist[0].shape[0]
        self.L = len(Alist)
        self.method = method
    
    def weight_single(self,A,label):
        cnt_in = 0
        sum_in = 0
        for i in range(self.k):
            idx = np.where(label==i)[0]
            sum_in += np.sum(A[idx, :][:, idx])
            cnt_in += len(idx)**2
        sum_out = np.sum(A) - sum_in
        cnt_out = self.n**2-cnt_in
        return (sum_in/cnt_in-sum_out/cnt_out)/(sum_in/cnt_in+(self.k-1)*sum_out/cnt_out)
    
    def weight_multi(self,init = False,label=[],n_init=10):
        w = np.empty(self.L)
        
        for i in range(len(self.Alist)):
            
            if init:
                evals, evecs = largest_eigsh(self.Alist[i].astype(float),self.k, which='LM')
                evals_abs = abs(evals)
                evecs = evecs[:,evals_abs.argsort()]
                evals = evals[evals_abs.argsort()]
                
                if self.method == 'gmm':
                    evecs_mean = np.zeros((self.k,self.k))
                    label = GaussianMixture(n_components=self.k,n_init=n_init).fit_predict(np.real(evecs[:,:]))
                else:
                    label = KMeans(n_clusters=self.k,n_init=n_init).fit_predict(np.real(evecs[:,:]))
    
        w[i] = self.weight_single(self.Alist[i],label)
        
        w = np.maximum(w,np.zeros(self.L))
        if np.sum(w)==0:
            w = np.random.randint(low=1,high=10,size=self.L)
        return w/np.sum(w)

    def wam(self,w,n_init=10):
        evals, evecs = largest_eigsh(sum([a*b for a,b in zip(w,self.Alist)]),self.k, which='LM')
        evals_abs = abs(evals)
        evecs = evecs[:,evals_abs.argsort()]
        evals = evals[evals_abs.argsort()]
        
        if self.method =='gmm':
            evecs_mean = np.zeros((self.k,self.k))
            label = GaussianMixture(n_components=self.k,n_init=n_init).fit_predict(np.real(evecs[:,:]))
        else:
            label = KMeans(n_clusters=self.k,n_init=n_init).fit_predict(np.real(evecs[:,:]))
        return label
    
    def opt(self,eps=1e-4,max_it=100,n_init=10):
        
        w_old = self.weight_multi(init=True)
        
        it = 0
        while it<max_it:
            it +=1
            label = self.wam(w_old)
            w_new = self.weight_multi(label=label,init=False,n_init=n_init)
            if sum((w_new-w_old)**2)<eps:
                break
            w_old = w_new
        
        return w_new,self.wam(w_new)


class MAM:
    def __init__(self,Alist,asso,k):
        self.Alist=Alist
        self.k = k
        self.n = Alist[0].shape[0]
        self.asso=asso
    
    def _Nam(self,comm):
        res = np.zeros((len(comm),len(comm)))
        for i in np.arange(len(comm)):
            for j in np.arange(i+1,len(comm)):
                if comm[i]==comm[j]:
                    res[i][j] = 1
        res = res+res.T
        np.fill_diagonal(res,1)
        return res
    
    def Mam(self,numini=5):
        NGM = []
        res = np.zeros((self.n,self.n))
        for index in np.arange(len(self.Alist)):
            ngm = float("inf")
            if self.asso[index]:
                ngm = -ngm
            for i in np.arange(numini):
                init = np.random.randint(low=0,high=self.k,size=self.n)
                comm,ngm_new = self._tabu_search(init,self.Alist[index],self.k,int(self.n/4), self.asso[index])
                if (self.asso[index] and ngm_new > ngm) or (not self.asso[index] and ngm_new < ngm):
                    ngm = ngm_new
                    comm_best = comm
            NGM.append(comm_best)
        for mod in NGM:
            res += self._Nam(mod)
        return (NGM,res)
    
    def _updateO(self,oldO,oldcommunity,indice,newlabel,Ab):
        oldlabel = oldcommunity[indice]
        newO = np.copy(oldO)
        newcommunity = np.copy(oldcommunity)
        newcommunity[indice] = newlabel
        posi=commFunc.position(newcommunity)
        changeO = sum(Ab[indice,posi[oldlabel]])
        changeN = sum(Ab[indice,posi[newlabel]])
        for j in np.arange(newcommunity.max()+1):
            change = sum(Ab[indice,posi[j]])
            newO[newlabel,j] = newO[newlabel,j]+change+change*(newlabel==j)-changeN*(oldlabel==j)
            newO[oldlabel,j] = newO[oldlabel,j]-change-change*(oldlabel==j)+changeO*(newlabel==j)
        newO[:,newlabel] = np.transpose(newO[newlabel,:])
        newO[:,oldlabel] = np.transpose(newO[oldlabel,:])
        return newO
    
    def _tabu_search(self,ini_community, Ab, k, tabu_size, asso=True, max_iterations=5000, max_stay=200, children=2):
        
        community = np.copy(ini_community)
        old_O = commFunc.O(community,Ab)
        ngm = sum(np.diagonal(old_O))-sum(sum(old_O)**2/np.sum(old_O))
        
        tabu_set = []
        iteration = 0
        stay = 0
        n = Ab.shape[0]
        
        while (iteration < max_iterations) and (stay < max_stay):  #  Stopping Criteria
            
            index =  random.randint(0,n-1) # Generate one randomly
            while index in tabu_set:
                index =  random.randint(0,n-1) # Generate another
            tabu_set = [index] + tabu_set[:tabu_size-1]
            stay = stay+1
            for label in np.setdiff1d(random.sample(range(0, self.k), children),community[index]):
                new_O = self._updateO(old_O,community,index,label,Ab)
                new_ngm = sum(np.diagonal(new_O))-sum(sum(new_O)**2/np.sum(new_O))
                
                if (asso and new_ngm > ngm) or (not asso and new_ngm < ngm):
                    stay = 0
                    old_O=new_O
                    community[index] = label
                    ngm = new_ngm
    
        iteration = iteration + 1
        return(community,ngm)

def ASC(Alist,k):
    res = np.zeros((Alist[0].shape[0],Alist[0].shape[0]))
    for Ab in Alist:
        if min(sum(Ab))+np.sum(Ab)/Ab.shape[0]>0:
            D = np.diag(np.squeeze(np.asarray(1/np.sqrt(sum(Ab)+np.sum(Ab)/Ab.shape[0]))))
        else:
            sigma = 1
            D = np.diag(np.squeeze(np.asarray(1/np.sqrt(sum(Ab)-min(sum(Ab))+sigma))))
        w, v = np.linalg.eig(np.matrix(D)*np.matrix(Ab)*np.matrix(D))
        w = np.array(list(map(abs,w)))
        X = v[:,w.argsort()[-k:][::-1]]
        X = X.real
        res += np.matmul(X,X.T)
    res = res/len(Alist)
    w, v = np.linalg.eig(res)
    w = np.array(list(map(abs,w)))
    X = v[:,w.argsort()[-k:][::-1]]
    kmeans = KMeans(n_clusters=k).fit(np.real(X))
    return(kmeans)


def lam_norm(in_list,out_list,k=2):
    w = [(x-y)/(x+(k-1)*y) for x,y in zip(in_list,out_list)]
    return([x/sum(w) for x in w])

def cond_num(in_list,out_list,k,n,rho):
    return sum([(x-y)/(x+(k-1)*y) for x,y in zip(in_list,out_list)])*n*rho/k


def genPPM(n,k,L,rho,pi,in_list,out_list):
    n_k = np.random.multinomial(n, pi, size=1)[0]
    gt = [n_k[i]*[i] for i in range(k)]
    gt = [elem for vec in gt for elem in vec]

    Ares = []
    for l in range(L):
        A = np.ones((k,k))*out_list[l]
        np.fill_diagonal(A,in_list[l])
        A = A*rho
        A_mean,A = genBer(A,n_k)
        Ares.append(A)
    
    return Ares,gt


def genSBM(n,k,L,rho,pi,Alist):
    n_k = np.random.multinomial(n, pi, size=1)[0]
    gt = [n_k[i]*[i] for i in range(k)]
    gt = [elem for vec in gt for elem in vec]
    
    Ares = []
    for l in range(L):
        A = Alist[l]*rho
        A_mean,A = genBer(A,n_k)
        Ares.append(A)
    
    return Ares,gt


def SC(matrix,k):
    evals, evecs = largest_eigsh(matrix,k,which='LM')
    evals_abs = abs(evals)
    evecs = evecs[:,evals_abs.argsort()]
    evals = evals[evals_abs.argsort()]
    model = KMeans(n_clusters=k).fit(np.real(evecs))
    label = model.predict(np.real(evecs))
    return label
