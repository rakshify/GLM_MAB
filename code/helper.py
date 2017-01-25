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
