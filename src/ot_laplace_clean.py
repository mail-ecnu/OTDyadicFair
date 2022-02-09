#!/usr/bin/env python

import numpy as np
from math import exp
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph as kn_graph

import ot
import scipy.optimize.linesearch  as ln
from cvxopt import matrix, spmatrix, solvers
from functools import reduce
from importlib import reload
import torch
import numba as nb

@nb.njit(fastmath=True,parallel=True)
def dist_arr_2(A, B, res):
    for i in nb.prange(A.shape[0]):
        for j in range(B.shape[0]):
            acc=0
            for k in range(A.shape[1]):
                acc+=(A[i,k]-B[j,k])**2
            res[i,j]=np.sqrt(acc)
    return res

solvers.options['show_progress'] = False
try:
    import compiled_mod

    reload(compiled_mod)
except:
    compiled_mod = None


def get_W(x, method='unif', param=None):
    """ returns the density estimation for a discrete distribution"""
    if method.lower() in ['rbf', 'gauss']:
        K = rbf_kernel(x, x, param)
        W = np.sum(K, 1)
        W = W / sum(W)
    else:
        if not method.lower() == 'unif':
            print("Warning: unknown density estimation, revert to uniform")
        W = np.ones(x.shape[0]) / x.shape[0]
    return W


def dots(*args):
    return reduce(np.dot, args)


def get_sim(x, sim, **kwargs):
    if sim == 'gauss':
        try:
            rbfparam = kwargs['rbfparam']
        except KeyError:
            rbfparam = 1 / (2 * (np.mean(ot.dist(x, x, 'sqeuclidean')) ** 2))
        S = rbf_kernel(x, x, rbfparam)
    elif sim == 'gaussthr':
        try:
            rbfparam = kwargs['rbfparam']
        except KeyError:
            rbfparam = 1 / (2 * (np.mean(ot.dist(x, x, 'sqeuclidean')) ** 2))
        try:
            thrg = kwargs['thrg']
        except KeyError:
            thrg = .5
        S = np.float64(rbf_kernel(x, x, rbfparam) > thrg)
    elif sim == 'knn':
        try:
            num_neighbors = kwargs['nn']
        except KeyError('sim="knn" requires the number of neighbors nn to be set'):
            num_neighbors = 3
        S = kn_graph(x, num_neighbors).toarray()
        S = (S + S.T) / 2
    return S


def compute_transport(xs, xt, method='lp', metric='euclidean', weights='unif', reg=0, solver=None, wparam=1, **kwargs):
    """
    Solve the optimal transport problem (OT)

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)

        s.t. \gamma 1 = \mu^s

             \gamma^T 1= \mu^t

             \gamma\geq 0
    where :

    - M is the metric cost matrix
    - Omega is the regularization term
    - mu_s and mu_t are the sample weights

    Parameters
    ----------
    xs : (ns x d) ndarray
        samples in the source domain
    xt : (nt x d) ndarray
        samples in the target domain
    reg: float()
        Regularization term >0
    method : str()
        Select the regularization term Omega

        - {'lp','None'} : classical optimization LP
            .. math::
                \Omega(\gamma)=0

        - {'qp'} : quadratic regularization
            .. math::
                \Omega(\gamma)=\|\gamma\|_F^2=\sum_{j,k}\gamma_{i,j}^2

        - {'sink', 'sinkhorn'} : sinkhorn (entropy) regularization
            .. math::
                \Omega(\gamma)=\sum_{i,j}\gamma_{i,j}\log(\gamma_{i,j})
        - {'laplace'} : laplacian regularization
            .. math::
                \Omega(\gamma)=(1-\alpha)/n_s^2\sum_{j,k}S_{i,j}\|T(\mathbf{x}^s_i)-T(\mathbf{x}^s_j)\|^2
                +\alpha/n_t^2\sum_{j,k}S_{i,j}^'\|T(\mathbf{x}^t_i)-T(\mathbf{x}^t_j)\|^2
            where the similarity matrices can be selected with the 'sim' parameter and 0<=alpha<=1 allow
            a fine tuning of the weights of each regularization.

            - sim='gauss'
                Gaussian kernel similarity with param 'rbfparam'
            - sim='gaussthr',rbfparam=1.,thrg=.5
                Gaussian kernel similarity with param 'rbfparam' and threshold 'thrg'
            - sim='gaussclass',rbfparam=1.,labels=y
                Gaussian kernel similarity with param 'rbfparam' intra-class only similarity
            - sim='knn',nn=3
                Knn similarity with param 'nn' (number of neighbors)
            - sim='knnclass',nn=3,labels=y
                Knn similarity with param 'nn' (number of neighbors) intra-class only similarity


        - {'laplace_traj'} : laplacian regularization on sample trajectory
            .. math::
                \Omega(\gamma)= (1-\alpha)/n_s^2\sum_{j,k}S_{i,j}\|T(\mathbf{x}^s_i)-\mathbf{x}^s_i-T(\mathbf{x}^s_j)+\mathbf{x}^s_j\|^2
                +\alpha/n_t^2\sum_{j,k}S_{i,j}^'\|T(\mathbf{x}^t_i)-\mathbf{x}^t_i-T(\mathbf{x}^t_j)+\mathbf{x}^t_j\|^2
            where the similarity matrices can be selected with the 'sim' parameter and 0<=alpha<=1 allow
            a fine tuning of the weights of each regularization.


    metric : str
        distance used for the computation of the M matrix.
        parameter can be:  'cityblock','cosine', 'euclidean',
        'sqeuclidean'.

        or any opf the distances described in the documentation of
        scipy.spatial.distance.cdist

    weights: str
        Defines the weights used for the source and target samples.


        Choose from:

        - {'unif'} :  uniform weights
            .. math::
                ,\quad\mu_k^t=1/n_t

        - {'gauss'} : gaussian kernel weights
            .. math::
                \mu_k^s=1/n_s\sum_j exp(-\|\mathbf{x}_k^s-\mathbf{x}_j^s\|^2/(2*wparam))

        - {'precomputed'} : user given weights
               then weightxs and weightxt should be given

    Returns
    -------
    gamma: (ns x nt) ndarray
        Optimal transportation matrix for the given parameters

    """

    # metric computation
    M = np.asarray(ot.dist(xs, xt, metric))

    # compute weights
    if weights.lower() == 'precomputed':
        w = kwargs['weightxs']
        wtest = kwargs['weightxt']
    else:
        w = get_W(xs, weights, wparam)
        wtest = get_W(xt, weights, wparam)

    # final if
    if method.lower() == 'lp':
        transp = ot.emd(w, wtest, M)
    elif method.lower() == 'qp':
        transp = computeTransportQP(w, wtest, M, reg, solver=solver)
    elif method.lower() in ['sink', 'sinkhorn']:
        transp = ot.sinkhorn(w, wtest, M, reg)
    elif method.lower() in ['laplace']:
        try:
            _ = kwargs['sim']
            alpha = kwargs['alpha']
        except KeyError:
            raise KeyError(
                'Method "laplace" require the similarity "sim" and the regularization term "alpha" to be passed as parameters')

        Ss = get_sim(xs, **kwargs)
        St = get_sim(xt, **kwargs)

        transp = computeTransportLaplacianSymmetric_fw(M, Ss, St, xs, xt, regls=reg * (1 - alpha), reglt=reg * alpha,
                                                       solver=solver, **kwargs)
    elif method.lower() in ['laplace_sinkhorn']:
        try:
            _ = kwargs['sim']
            alpha = kwargs['alpha']
            eta = kwargs['eta']
        except KeyError:
            raise KeyError(
                'Method "laplace_sinkhorn" requires the similarity "sim", the regularization terms "eta" and "alpha" to be passed as parameters')

        Ss = get_sim(xs, **kwargs)
        St = get_sim(xt, **kwargs)

        transp = computeTransportLaplacianSymmetric_fw_sinkhorn(M, Ss, St, xs, xt, reg = reg,
                                                                regls=eta * (1 - alpha), reglt=eta * alpha, **kwargs)
    elif method.lower() in ['laplace_traj']:
        try:
            _ = kwargs['sim']
            alpha = kwargs['alpha']
        except KeyError:
            raise KeyError(
                'Method "laplace_traj" require the similarity "sim" and the regularization term "alpha" to be passed as parameters')

        Ss = get_sim(xs, **kwargs)
        St = get_sim(xt, **kwargs)

        transp = computeTransportLaplacianSymmetricTraj_fw(M, Ss, St, xs, xt, regls=reg * (1 - alpha),
                                                           reglt=reg * alpha, solver=solver, **kwargs)

    else:
        print('Warning: unknown method {method}. Fallback to LP'.format(method=method))
        transp = ot.emd(w, wtest, M)

    return transp

def myreg_compute_transport(xs, xt, pis, pit, method='lp', metric='euclidean', weights='unif', reg=0, solver=None, wparam=1, **kwargs):
    """
    Solve the optimal transport problem (OT)

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)

        s.t. \gamma 1 = \mu^s

             \gamma^T 1= \mu^t

             \gamma\geq 0
    where :

    - M is the metric cost matrix
    - Omega is the regularization term
    - mu_s and mu_t are the sample weights

    Parameters
    ----------
    xs : (ns x d) ndarray
        samples in the source domain
    xt : (nt x d) ndarray
        samples in the target domain
    reg: float()
        Regularization term >0
    method : str()
        Select the regularization term Omega

        - {'lp','None'} : classical optimization LP
            .. math::
                \Omega(\gamma)=0

        - {'qp'} : quadratic regularization
            .. math::
                \Omega(\gamma)=\|\gamma\|_F^2=\sum_{j,k}\gamma_{i,j}^2

        - {'sink', 'sinkhorn'} : sinkhorn (entropy) regularization
            .. math::
                \Omega(\gamma)=\sum_{i,j}\gamma_{i,j}\log(\gamma_{i,j})
        - {'laplace'} : laplacian regularization
            .. math::
                \Omega(\gamma)=(1-\alpha)/n_s^2\sum_{j,k}S_{i,j}\|T(\mathbf{x}^s_i)-T(\mathbf{x}^s_j)\|^2
                +\alpha/n_t^2\sum_{j,k}S_{i,j}^'\|T(\mathbf{x}^t_i)-T(\mathbf{x}^t_j)\|^2
            where the similarity matrices can be selected with the 'sim' parameter and 0<=alpha<=1 allow
            a fine tuning of the weights of each regularization.

            - sim='gauss'
                Gaussian kernel similarity with param 'rbfparam'
            - sim='gaussthr',rbfparam=1.,thrg=.5
                Gaussian kernel similarity with param 'rbfparam' and threshold 'thrg'
            - sim='gaussclass',rbfparam=1.,labels=y
                Gaussian kernel similarity with param 'rbfparam' intra-class only similarity
            - sim='knn',nn=3
                Knn similarity with param 'nn' (number of neighbors)
            - sim='knnclass',nn=3,labels=y
                Knn similarity with param 'nn' (number of neighbors) intra-class only similarity


        - {'laplace_traj'} : laplacian regularization on sample trajectory
            .. math::
                \Omega(\gamma)= (1-\alpha)/n_s^2\sum_{j,k}S_{i,j}\|T(\mathbf{x}^s_i)-\mathbf{x}^s_i-T(\mathbf{x}^s_j)+\mathbf{x}^s_j\|^2
                +\alpha/n_t^2\sum_{j,k}S_{i,j}^'\|T(\mathbf{x}^t_i)-\mathbf{x}^t_i-T(\mathbf{x}^t_j)+\mathbf{x}^t_j\|^2
            where the similarity matrices can be selected with the 'sim' parameter and 0<=alpha<=1 allow
            a fine tuning of the weights of each regularization.


    metric : str
        distance used for the computation of the M matrix.
        parameter can be:  'cityblock','cosine', 'euclidean',
        'sqeuclidean'.

        or any opf the distances described in the documentation of
        scipy.spatial.distance.cdist

    weights: str
        Defines the weights used for the source and target samples.


        Choose from:

        - {'unif'} :  uniform weights
            .. math::
                ,\quad\mu_k^t=1/n_t

        - {'gauss'} : gaussian kernel weights
            .. math::
                \mu_k^s=1/n_s\sum_j exp(-\|\mathbf{x}_k^s-\mathbf{x}_j^s\|^2/(2*wparam))

        - {'precomputed'} : user given weights
               then weightxs and weightxt should be given

    Returns
    -------
    gamma: (ns x nt) ndarray
        Optimal transportation matrix for the given parameters

    """

    # metric computation
    M = np.asarray(ot.dist(xs, xt, metric))

    # compute weights


    transp = myreg_computeTransportSymmetric_fw(M, xs, xt, pi_s, pi_t, regls=reg * (1 - alpha),
                                                           reglt=reg * alpha, solver=solver, **kwargs)

    return transp



def get_K_laplace(Nini, Nfin, S, Sigma):
    """
    fonction pas efficace mais qui marche
    """

    K = np.zeros((Nini * Nfin, Nini * Nfin))

    def idx(i, j):
        return np.ravel_multi_index((i, j), (Nini, Nfin))

    # contruction de K!!!!!!
    for i in range(Nini):
        for j in range(Nini):
            for k in range(Nfin):
                for l in range(Nfin):
                    K[idx(i, k), idx(i, l)] += S[i, j] * Sigma[k, l]
                    K[idx(j, k), idx(j, l)] += S[i, j] * Sigma[k, l]
                    K[idx(i, k), idx(j, l)] += -2 * S[i, j] * Sigma[k, l]
    return K


def get_K_laplace2(Nini, Nfin, St, Sigmat):
    """
    fonction pas efficace mais qui marche
    """
    K = np.zeros((Nini * Nfin, Nini * Nfin))

    def idx(i, j):
        return np.ravel_multi_index((i, j), (Nini, Nfin))

    # contruction de K!!!!!!
    for i in range(Nfin):
        for j in range(Nfin):
            for k in range(Nini):
                for l in range(Nini):
                    K[idx(k, i), idx(l, i)] += St[i, j] * Sigmat[k, l]
                    K[idx(k, j), idx(l, j)] += St[i, j] * Sigmat[k, l]
                    K[idx(k, i), idx(l, j)] += -2 * St[i, j] * Sigmat[k, l]
    return K




def get_gradient1(L, X, transp):
    """
    Compute gradient for the laplacian reg term on transported sources
    """
    return np.dot(L + L.T, np.dot(transp, np.dot(X, X.T)))


def get_gradient2(L, X, transp):
    """
    Compute gradient for the laplacian reg term on transported targets
    """
    return np.dot(X, np.dot(X.T, np.dot(transp, L + L.T)))


def get_laplacian(S):
    L = np.diag(np.sum(S, axis=0)) - S
    return L


def quadloss(transp, K):
    """
    Compute quadratic loss with matrix K
    """
    return np.sum(transp.flatten() * np.dot(K, transp.flatten()))


def quadloss1(transp, L, X):
    """
    Compute loss for the laplacian reg term on transported sources
    """
    return np.trace(np.dot(X.T, np.dot(transp.T, np.dot(L, np.dot(transp, X)))))


def quadloss2(transp, L, X):
    """
    Compute loss for the laplacian reg term on transported sources
    """
    return np.trace(np.dot(X.T, np.dot(transp, np.dot(L, np.dot(transp.T, X)))))


### ------------------------------- Optimal Transport ---------------------------------------


def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]


########### Compute transport with a QP solver

def computeTransportQP(distribS, distribT, distances, reg=0, K=None, solver=None):
    # init data
    Nini = len(distribS)
    Nfin = len(distribT)

    # generate probability distribution of each class
    p1p2 = np.concatenate((distribS, distribT))
    p1p2 = p1p2[0:-1]
    # generate cost matrix
    costMatrix = distances.flatten()

    # express the constraints matrix
    I = []
    J = []
    for i in range(Nini):
        for j in range(Nfin):
            I.append(i)
            J.append(i * Nfin + j)
    for i in range(Nfin - 1):
        for j in range(Nini):
            I.append(i + Nini)
            J.append(j * Nfin + i)

    A = spmatrix(1.0, I, J)

    # positivity condition
    G = spmatrix(-1.0, range(Nini * Nfin), range(Nini * Nfin))
    if not K == None:
        P = matrix(K) + reg * spmatrix(1.0, range(Nini * Nfin), range(Nini * Nfin))
    else:
        P = reg * spmatrix(1.0, range(Nini * Nfin), range(Nini * Nfin))

    sol = solvers.qp(P, matrix(costMatrix), G, matrix(np.zeros(Nini * Nfin)), A, matrix(p1p2), solver=solver)
    S = np.array(sol['x'])

    transp = np.reshape([l[0] for l in S], (Nini, Nfin))
    return transp


def computeTransportLaplacianSymmetric(distances, Ss, St, xs, xt, reg=0, regls=0, reglt=0, solver=None):
    distribS = np.ones((xs.shape[0], 1)) / xs.shape[0]
    distribT = np.ones((xt.shape[0], 1)) / xt.shape[0]

    Nini = len(distribS)
    Nfin = len(distribT)

    Sigmat = np.dot(xt, xt.T)
    Sigmas = np.dot(xs, xs.T)

    # !!!! MArche pas a refaire je me suis plante dans le deuxieme K
    if compiled_mod == None:
        Ks = get_K_laplace(Nini, Nfin, Ss, Sigmat)
        Kt = get_K_laplace2(Nini, Nfin, St, Sigmas)
    else:
        Ks = compiled_mod.get_K_laplace(Nini, Nfin, Ss, Sigmat)
        Kt = compiled_mod.get_K_laplace2(Nini, Nfin, St, Sigmas)

    K = (Ks * regls + Kt * reglt)

    transp = computeTransportQP(distribS, distribT, distances, reg=reg, K=K, solver=solver)

    # print "loss:",np.sum(transp*distances)+quadloss(transp,K)/2

    return transp


def computeTransportLaplacianSource(distances, Ss, xs, xt, reg=0, regl=0, solver=None):
    distribS = np.ones((xs.shape[0], 1)) / xs.shape[0]
    distribT = np.ones((xt.shape[0], 1)) / xt.shape[0]

    Nini = len(distribS)
    Nfin = len(distribT)

    Sigma = np.dot(xt, xt.T)

    if compiled_mod == None:
        K = get_K_laplace(Nini, Nfin, Ss, Sigma)
    else:
        K = compiled_mod.get_K_laplace(Nini, Nfin, Ss, Sigma)

    K = K * regl

    transp = computeTransportQP(distribS, distribT, distances, reg=reg, K=K, solver=solver)
    return transp


def computeTransportLaplacianSource_fw(distances, Ss, xs, xt, reg=0, regl=0, solver=None, nbitermax=400, thr_stop=1e-6,
                                       step='opt'):
    distribS = np.ones((xs.shape[0],)) / xs.shape[0]
    distribT = np.ones((xt.shape[0],)) / xt.shape[0]

    # compute laplacian
    L = get_laplacian(Ss)

    loop = True

    transp = ot.emd(distribS, distribT, distances)
    # transp=np.dot(distribS,distribT.T)

    niter = 0
    while loop:

        old_transp = transp.copy()

        # G=get_gradient(old_transp,K)
        G = regl * get_gradient1(L, xt, old_transp)

        transp0 = ot.emd(distribS, distribT, distances + G)

        E = transp0 - old_transp
        # Ge

        if step == 'opt':
            # optimal step size !!!
            tau = max(0, min(1, (-np.sum(E * distances) - np.sum(E * G)) / (2 * regl * quadloss1(E, L, xt))))
        else:
            # other step size just in case
            tau = 2. / (niter + 2)
        # print "tau:",tau

        transp = old_transp + tau * E

        if niter >= nbitermax:
            loop = False

        if np.sum(np.abs(transp - old_transp)) < thr_stop:
            loop = False
        # print niter

        niter += 1

    return transp


def computeTransportLaplacianSymmetric_fw(distances, Ss, St, xs, xt, reg=1e-9, regls=0, reglt=0, solver=None,
                                          nbitermax=400, thr_stop=1e-8, step='opt', **kwargs):
    distribS = np.ones((xs.shape[0],)) / xs.shape[0]
    distribT = np.ones((xt.shape[0],)) / xt.shape[0]

    Ls = get_laplacian(Ss)
    Lt = get_laplacian(St)

    loop = True

    transp = ot.emd(distribS, distribT, distances)

    niter = 0
    while loop:

        old_transp = transp.copy()

        G = np.asarray(regls * get_gradient1(Ls, xt, old_transp) + reglt * get_gradient2(Lt, xs, old_transp))

        transp0 = ot.emd(distribS, distribT, distances + G)

        E = transp0 - old_transp
        # Ge=get_gradient(E,K)

        if step == 'opt':
            # optimal step size !!!
            tau = max(0, min(1, (-np.sum(E * distances) - np.sum(E * G)) / (
                        2 * regls * quadloss1(E, Ls, xt) + 2 * reglt * quadloss2(E, Lt, xs))))
        else:
            # other step size just in case
            tau = 2. / (niter + 2)  # print "tau:",tau

        transp = old_transp + tau * E

        # print "loss:",np.sum(transp*distances)+quadloss(transp,K)/2

        if niter >= nbitermax:
            loop = False

        err = np.sum(np.abs(transp - old_transp))

        if err < thr_stop:
            loop = False
        # print niter

        niter += 1

        if niter % 1000 == 0:
            print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(niter, err))

    # print "loss:",np.sum(transp*distances)+quadloss(transp,K)/2

    return transp

def get_symmetric_reg_grad(x_0, x_1, idx_p0, idx_p1, gamma):
    x_0, x_1 = torch.Tensor(x_0), torch.Tensor(x_1)
    gamma = torch.Tensor(gamma)
    gamma.requires_grad_()
    n0, d0 = x_0.shape
    n1, d1 = x_1.shape
    pi_0 = n0 / (n0 + n1)
    pi_1 = 1 - pi_0


    x_0_rep = pi_0 * x_0 + n0 * pi_1 * torch.mm(gamma, x_1)
    x_1_rep = pi_1 * x_1 + n1 * pi_0 * torch.mm(gamma.t(), x_0)
    

    new_x = torch.zeros((x_0.shape[0]+x_1.shape[0], x_0.shape[1]))
    new_x[idx_p0, :] = x_0_rep
    new_x[idx_p1, :] = x_1_rep
    m = torch.nn.ReLU6()
    new_x = m((new_x - 0.19) * 10000)
#     new_x = new_x.ge(0.5)
    regs = new_x - new_x.t()
#     print(new_x.sum())
#     print(regs.sum())
#     import pdb; pdb.set_trace()
    return torch.sum(regs*regs).detach().numpy(), torch.autograd.grad(torch.sum(regs*regs), gamma)[0].detach().numpy()
    
    
def myreg_computeTransportSymmetric_fw(distances, xs, xt, idx_ps, idx_pt, reg=1e-9, solver=None,
                                          nbitermax=400, thr_stop=1e-8, step='opt', **kwargs):
    distribS = np.ones((xs.shape[0],)) / xs.shape[0]
    distribT = np.ones((xt.shape[0],)) / xt.shape[0]

    loop = True

    transp = ot.emd(distribS, distribT, distances)

    niter = 0
    while loop:

        old_transp = transp.copy()

        L, gradL = get_symmetric_reg_grad(xs, xt, idx_ps, idx_pt, old_transp)
        G = np.asarray(reg * gradL)

        transp0 = ot.emd(distribS, distribT, distances + G)

        E = transp0 - old_transp
        # Ge=get_gradient(E,K)

            # other step size just in case
        tau = 2. / (niter + 2)  # print "tau:",tau

        transp = old_transp + tau * E

        # print "loss:",np.sum(transp*distances)+quadloss(transp,K)/2

        if niter >= nbitermax:
            loop = False

        err = np.sum(np.abs(transp - old_transp))

        if err < thr_stop:
            loop = False
        # print niter

        niter += 1

        if niter % 1000 == 0:
            print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(niter, err))

    # print "loss:",np.sum(transp*distances)+quadloss(transp,K)/2

    return transp

def computeTransportLaplacianSymmetric_fw_sinkhorn(distances, Ss, St, xs, xt, reg=1e-9, regls=0, reglt=0, nbitermax=400,
                                                   thr_stop=1e-8, **kwargs):
    distribS = np.ones((xs.shape[0], 1)) / xs.shape[0]
    distribS = distribS.ravel()
    distribT = np.ones((xt.shape[0], 1)) / xt.shape[0]
    distribT = distribT.ravel()

    Ls = get_laplacian(Ss)
    Lt = get_laplacian(St)

    maxdist = np.max(distances)

    regmax = 300. / maxdist
    reg0 = regmax * (1 - exp(-reg / regmax))

    transp = ot.sinkhorn(distribS, distribT, distances, reg)

    niter = 1
    while True:
        old_transp = transp.copy()
        G = np.asarray(regls * get_gradient1(Ls, xt, old_transp) + reglt * get_gradient2(Lt, xs, old_transp))
        transp0 = ot.sinkhorn(distribS, distribT, distances + G, reg)
        E = transp0 - old_transp

        # do a line search for best tau
        def f(tau):
            T = (1 - tau) * old_transp + tau * transp0
            # print np.sum(T*distances),-1./reg0*np.sum(T*np.log(T)),regls*quadloss1(T,Ls,xt),reglt*quadloss2(T,Lt,xs)
            return np.sum(T * distances) + 1. / reg0 * np.sum(T * np.log(T)) + \
                   regls * quadloss1(T, Ls, xt) + reglt * quadloss2(T, Lt, xs)

        # compute f'(0)
        res = regls * (np.trace(np.dot(xt.T, np.dot(E.T, np.dot(Ls, np.dot(old_transp, xt))))) + \
                       np.trace(np.dot(xt.T, np.dot(old_transp.T, np.dot(Ls, np.dot(E, xt)))))) \
              + reglt * (np.trace(np.dot(xs.T, np.dot(E, np.dot(Lt, np.dot(old_transp.T, xs))))) + \
                         np.trace(np.dot(xs.T, np.dot(old_transp, np.dot(Lt, np.dot(E.T, xs))))))

        # derphi_zero = np.sum(E*distances) - np.sum(1+E*np.log(old_transp))/reg + res
        derphi_zero = np.sum(E * distances) + np.sum(E * (1 + np.log(old_transp))) / reg0 + res

        tau, cost = ln.scalar_search_armijo(f, f(0), derphi_zero, alpha0=0.99)

        if tau is None:
            break
        transp = (1 - tau) * old_transp + tau * transp0

        err = np.sum(np.fabs(E))

        if niter >= nbitermax or  err < thr_stop:
            break
        niter += 1

        if niter % 100 == 0:
            print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(niter, err))

    return transp


def computeTransportLaplacianSymmetricTraj_fw(distances, Ss, St, xs, xt, reg=0, regls=0, reglt=0, solver=None,
                                              nbitermax=400, thr_stop=1e-8, step='opt', **kwargs):
    distribS = np.ones((xs.shape[0],)) / xs.shape[0]
    distribT = np.ones((xt.shape[0],)) / xt.shape[0]

    ns = xs.shape[0]
    nt = xt.shape[0]

    Ls = get_laplacian(Ss)
    Lt = get_laplacian(St)

    Cs = np.asarray(-regls / ns * dots(Ls + Ls.T, xs, xt.T))
    Ct = np.asarray(-reglt / nt * dots(xs, xt.T, Lt + Lt.T))

    loop = True

    transp = ot.emd(distribS, distribT, distances)

    Ctot = distances + Cs + Ct

    niter = 0
    while loop:

        old_transp = transp.copy()

        G = np.asarray(regls * get_gradient1(Ls, xt, old_transp) + reglt * get_gradient2(Lt, xs, old_transp))

        transp0 = ot.emd(distribS, distribT, Ctot + G)

        E = transp0 - old_transp
        # Ge=get_gradient(E,K)

        if step == 'opt':
            # optimal step size !!!
            tau = max(0, min(1, (-np.sum(E * Ctot) - np.sum(E * G)) / (2 * regls * quadloss1(E, Ls, xt)
                                                                       + 2 * reglt * quadloss2(E, Lt, xs))))
        else:
            # other step size just in case
            tau = 2. / (niter + 2)  # print "tau:",tau

        transp = old_transp + tau * E

        # print "loss:",np.sum(transp*distances)+quadloss(transp,K)/2

        if niter >= nbitermax:
            loop = False

        err = np.sum(np.abs(transp - old_transp))
        if  err < thr_stop:
            loop = False
        # print niter

        niter += 1

        if niter % 100 == 0:
            print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(niter, err))

    # print "loss:",np.sum(transp*distances)+quadloss(transp,K)/2

    return transp


def multi_computeTransportSymmetric_fw(distances, xs, xt, weight, reg=1e-9, solver=None,
                                          nbitermax=400, thr_stop=1e-8, step='opt', **kwargs):
    distribS = np.ones((xs.shape[0],)) / xs.shape[0]
    distribT = np.ones((xt.shape[0],)) / xt.shape[0]

    loop = True

    transp = ot.emd(distribS, distribT, distances)

    niter = 0
    while loop:

        old_transp = transp.copy()
        distances = ot.utils.dist(xs, xt)
        L, gradL = get_multi_symmetric_reg_grad(xs, xt, weight, old_transp)
        G = np.asarray(reg * gradL)
        
        # transp0 = ot.emd(distribS, distribT, distances + G, numItermax=5e3)
        transp0 = ot.emd(distribS, distribT, distances + G)

        E = transp0 - old_transp
        # Ge=get_gradient(E,K)

            # other step size just in case
        tau = 2. / (niter + 2)  # print "tau:",tau

        transp = old_transp + tau * E

        # print "loss:",np.sum(transp*distances)+quadloss(transp,K)/2

        if niter >= nbitermax:
            loop = False

        err = np.sum(np.abs(transp - old_transp))

        if err < thr_stop:
            loop = False
        # print niter

        niter += 1

        if niter % 1000 == 0:
            print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(niter, err))

    # print "loss:",np.sum(transp*distances)+quadloss(transp,K)/2

    return transp

def get_multi_symmetric_reg_grad(x_0, x_1,weight,gamma):
    x_0, x_1 = torch.Tensor(x_0).cuda(), torch.Tensor(x_1).cuda()
    gamma = torch.Tensor(gamma).cuda()
    gamma.requires_grad_()
    k = x_0.shape[0]
    d = x_0.shape[1]
    
    b = torch.ones((k,)).cuda()/k
    
    X = x_0 + weight * torch.reshape(1. / b, (-1, 1)) * torch.matmul(gamma, x_1)
    m = torch.nn.ReLU6()
    new_x = m((X - 0.19) * 10000)
#     new_x = new_x.ge(0.5)
    regs = new_x - new_x.t()
#     print(new_x.sum())
#     print(regs.sum())
#     import pdb; pdb.set_trace()
    return torch.sum(regs*regs).detach().cpu().numpy(), torch.autograd.grad(torch.sum(regs*regs), gamma)[0].detach().cpu().numpy()

def multi_node_computeTransportSymmetric_fw(xs, xt, xt2, weight,method='laplace', reg=1e-9, reg1=1, reg2=1, solver=None, wparam=1,nbitermax=400, thr_stop=1e-8):
    distribS = np.ones((xs.shape[0],)) / xs.shape[0]
    distribT = np.ones((xt.shape[0],)) / xt.shape[0]  
    d1, d2 = xt.shape[1], xt2.shape[1]
    loop = True
    import time 
    start = time.time()
    myx = np.concatenate([xt, xt2], axis=-1)
    myres = np.empty((xs.shape[0],myx.shape[0]))
    M_i = dist_arr_2(xs, myx, myres)
    # M_i = reg1*ot.utils.dist(xs[:,:d1], xt) + reg2*ot.utils.dist(xs[:,d1:], xt2)
    print(f'dist clac lasted {time.time()-start}')
    start = time.time()
    transp = ot.emd(distribS, distribT, M_i)
    print(f'emd clac lasted {time.time()-start}')


    niter = 0
    start = time.time()
    while loop:

        old_transp = transp.copy()
        start0 = time.time()
        # L, gradL = get_node_multi_symmetric_reg_grad(xs, xt, xt2, reg1,reg2, weight, old_transp)
        # G = np.asarray(reg * gradL)
        # M_i = reg1*ot.utils.dist(xs[:,:d1], xt) + reg2*ot.utils.dist(xs[:,d1:], xt2)
        myx = np.concatenate([reg1*xt, reg2*xt2], axis=-1)
        myres = np.empty((xs.shape[0],myx.shape[0]))
        myxs = np.ones_like(xs) 
        myxs[:d1] = reg1 * xs[:d1]
        myxs[d1:] = reg2 * xs[d1:]
        M_i = dist_arr_2(xs, myx, myres)
        start0 = time.time()
        # transp0 = ot.emd(distribS, distribT, distances + G, numItermax=5e3)
        # transp0 = ot.emd(distribS, distribT, M_i + G)
        transp0 = ot.emd(distribS, distribT, M_i)
#        print(f'emd clac lasted {time.time()-start0}')
        E = transp0 - old_transp
        # Ge=get_gradient(E,K)

            # other step size just in case
        tau = 2. / (niter + 2)  # print "tau:",tau

        transp = old_transp + tau * E

        # print "loss:",np.sum(transp*distances)+quadloss(transp,K)/2

        if niter >= nbitermax:
            loop = False

        err = np.sum(np.abs(transp - old_transp))

        if err < thr_stop:
            loop = False
        # print niter

        niter += 1

        if niter % 1000 == 0:
            print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(niter, err))

    # print "loss:",np.sum(transp*distances)+quadloss(transp,K)/2
    print(f'update {niter} times and lasted {time.time()-start} seconds')
    return transp

def get_node_multi_symmetric_reg_grad(xs, xt, xt2, reg1,reg2, weight, gamma):
    xs, xt, xt2 = torch.Tensor(xs).cuda(), torch.Tensor(xt).cuda(), torch.Tensor(xt2).cuda()
    gamma = torch.Tensor(gamma).cuda()
    gamma.requires_grad_()
    k = xs.shape[0]
    d = xs.shape[1]
    d1 = xt.shape[1]
    
    b = torch.ones((k,)).cuda()/k
    
    X = xs[:, :d1]+ weight * torch.reshape(1. / b, (-1, 1)) * torch.matmul(gamma, xt)
    m = torch.nn.ReLU6()
    new_x = m((X - 0.19) * 10000)
#     new_x = new_x.ge(0.5)
    regs = new_x - new_x.t()
#     print(new_x.sum())
#     print(regs.sum())
#     import pdb; pdb.set_trace()
    return torch.sum(regs*regs).detach().cpu().numpy(), torch.autograd.grad(torch.sum(regs*regs), gamma)[0].detach().cpu().numpy()