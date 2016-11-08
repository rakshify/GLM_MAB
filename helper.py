import math, numpy as np



class Lin_Alg:
    '''
        CLASS CONTAINING STATIC LINEAR ALGEBRA HELPER FUNCTIONS
    '''

    @staticmethod
    def update_mat_inv(M_inv, X):
        '''
           FUNCTION TO GET DYNAMICALLY CHANGING MATRIX'S INVERSE

           ARGUMENTS: 2
        	M_inv  : Current matrix inverse
        	add    : Change done in the matrix

           RETURNS: NUMPY ARRAY
        	Updated inverse of the matrix
        '''
    	c = 1 / (1 + np.dot(np.dot(X, M_inv), X.transpose()))
        add = np.outer(X, X.transpose())
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
    else:
        return 0



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
		# print val
		return math.exp(val) / (1 + math.exp(val))
