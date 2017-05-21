import math, scipy, numpy as np



class Lin_Alg:
    '''
        CLASS CONTAINING STATIC LINEAR ALGEBRA HELPER FUNCTIONS
    '''

    @staticmethod
    def update_mat_inv(M_inv, X, mu = None):
        '''
           FUNCTION TO GET DYNAMICALLY CHANGING MATRIX'S INVERSE

           ARGUMENTS: 2
        	M_inv  : Current matrix inverse
        	add    : Change done in the matrix

           RETURNS: NUMPY ARRAY
        	Updated inverse of the matrix
        '''
        u = X
        v = X
        if mu != None:
            u = mu * X
            v = (1 - mu) * X
        c = 1 / (1 + np.dot(np.dot(u, M_inv), v.transpose()))
        add = np.outer(u, v.transpose())
    	inter = c * np.dot(np.dot(M_inv, add), M_inv)
    	return M_inv - inter



'''
   LINK FUNCTION FOR GLM

   ARGUMENTS: 2
	name	: Name of the link function
	*argv	: Set of arguments required for the aforementioned link function

   RETURNS: FLOAT
	Real valued result of the link function
'''
def link_func(name, val):
    if name == "logistic":
        return logistic(val)
    elif name == "identity":
        return val
    else:
        return 0


'''
   FUNCTION TO GET SAMPLES FROM MULTIVARIATE GAUSSIAN

   ARGUMENTS: 3
	mean        : Mean of the Gaussian distribution
    covariance  : Covariance of the Gaussian distribution
    samples     : Number of samples to be drawn

   RETURNS: FLOAT
	The output of logistic function
'''
def gaussian(mean, covariance, samples):
    # print "asked samples = ",samples
    nf = mean.shape[0]
    shape = [samples]
    final_shape = list(shape[:])
    final_shape.append(nf)
    x = np.random.standard_normal(final_shape).reshape(-1, nf)
    # sqc = np.nan_to_num(np.sqrt(covariance))
    sqc = scipy.linalg.sqrtm(covariance)
    print "covariance params = ", np.amax(sqc), np.amin(sqc)
    return np.dot(x, sqc) + mean


'''
   LOGISTIC FUNCTION

   ARGUMENTS: 1
	val: input for logistic function

   RETURNS: FLOAT
	The output of logistic function
'''
def logistic(val):
	try:
		return 1 / (1 + math.exp(-val))
	except:
		return math.exp(val) / (1 + math.exp(val))


def func(w, contexts, rewards):
    l = len(contexts)
    val = np.zeros(contexts[0].shape[0])
    for i in range(l):
        mul = 0
        if rewards[i] == 1:
            v = -np.dot(w, contexts[i])
            mul = logistic(v)
        else:
            v = np.dot(w, contexts[i])
            mul = -logistic(v)
        # print contexts[i].shape
        # print "mul = ", mul
        val += mul * contexts[i]
        # print val.shape
    return val


def gd(w, Q, contexts, rewards, alpha):
    feats = w.shape[0]
    tw = np.zeros(feats)
    for i in range(feats):
        tw[i] = w[i]
    for i in range(1000):
        eta = 1 / np.sqrt(float(i + 2))
        prior = alpha * np.dot(Q, (tw - w))
        likeli = func(w, contexts, rewards)
        grad = prior + likeli
        # if np.linalg.norm(grad) <= 0.001:
        #     break
        # print tw
        tw = tw - eta * grad
        # print tw
        # print np.linalg.norm(tw)
        tw /= np.linalg.norm(tw)
        # print tw
        # print eta * grad
        # print likeli
        # print prior
        # print "==============="
        # print tw
        # print contexts[0]
        # print np.linalg.norm(contexts[0]), np.linalg.norm(tw), np.dot(tw, contexts[0])
        # print "==============="
        # print "==============="
        if np.linalg.norm(eta * grad) < 0.001:
            print "i = ", i
            if i == 0:
                print grad
                print likeli
                print prior
                print np.linalg.norm(contexts[0]), np.linalg.norm(tw), np.dot(tw, contexts[0])
                print "==============="
            break
    return tw


def sgld(w, f, Q, contexts, rewards, alpha, N = 100.0):
    feats = w.shape[0]
    tw = np.zeros(feats)
    for i in range(feats):
        tw[i] = w[i]
    
    for i in range(100):
        eta = 1.0 / np.sqrt(float(i + 2))
        grad = (alpha / 2.0) * np.dot(Q, (tw - w))
        grad -= func(w, contexts, rewards) * N / float(len(contexts))
        # if np.linalg.norm(grad) <= 0.001:
        #     break
        eps = np.random.normal(0, eta)
        tw = tw + eta * grad / 2.0 + eps

    return tw


# class SGLD():
#     def __init__(self, feats):
#         self.f = np.zeros(feats)
#
#     def sgld(self, w, Q, contexts, rewards, alpha, N = 100):
#         n, f = contexts.shape
