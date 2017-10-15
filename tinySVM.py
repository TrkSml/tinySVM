import numpy as np 
import sys

from cvxopt import matrix
from cvxopt import solvers

solvers.options['show_progress'] = False

def Linear(x1,x2):
	return np.dot(x1,x2.T)

def Polynomial(x1,x2,d=2,C=0):
 	return np.power(np.dot(x1,x2.T)+C,d)

def RBF(x1,x2,sigma=1):
	return np.exp(-np.linalg.norm(x1-x2)**2/(2*sigma**2))

def Hyperbolic_Tangent(x1,x2,gamma=1,C=0):
	return np.tanh(gamma*np.dot(x1,x2.T)+C)

def test_kernel(kernel):
	if kernel:
		if not kernel in ['linear','poly','rbf','tanh']:
			print 'Error: Unable to perform classification :\n\
			The kernel must be either \"linear\",\'poly\',\'rbf\' or \'tanh\'\
			'
			sys.exit(0)

def construct_kernel(X,kernel=None):

	if kernel:
		Kernel=np.zeros((X.shape[0],X.shape[0]))

		if kernel=="rbf":
			for i,x_i in enumerate(X):
				for j,x_j in enumerate(X):
					Kernel[i,j]=RBF(x_i,x_j)

		if kernel=="tanh":
			for i,x_i in enumerate(X):
				for j,x_j in enumerate(X):
					Kernel[i,j]=Hyperbolic_Tangent(x_i,x_j)

		if kernel=="poly":
			for i,x_i in enumerate(X):
				for j,x_j in enumerate(X):
					Kernel[i,j]=Polynomial(x_i,x_j)
		return Kernel
		
	else:
		return np.dot(X,X.T)


class SVM:

	def __init__(self,X,Y,kernel=None,margin=0):
		self.X=X
		self.Y=Y
		self.kernel=test_kernel(kernel)
		self.margin=margin
				

	def construct_model(self):

		if not (type(self.X).__module__ == np.__name__ or type(self.Y).__module__ == np.__name__):
			print "Error: Either input or output uknown type .. please make \
			sure you are taking numpy array for both."
			sys.exit(0)

		number_of_features=self.X.shape[1]
		number_of_samples=self.X.shape[0]
		total_set=set(list(self.Y))
		total_list=list(total_set)

		if not len(set(list(self.Y)))<=2:
			print "The target must contain two values .. Sorry :("
			sys.exit(0)

		self.dict={}
		if total_list[0]>0:
			self.dict[1]=total_list[0]
			if len(total_list)>1:
				self.dict[-1]=total_list[1]
		else:
			if len(total_list)>1:
				self.dict[-1]=total_list[1]
				self.dict[1]=total_list[1]
			else:
				self.dict[-1]=total_list[0]

		matrix_X=construct_kernel(X,self.kernel)
		matrix_Y=np.dot(self.Y.T,self.Y)

		K=np.dot(matrix_Y,matrix_X)

		P = matrix(K,tc='d')
		q = matrix(-np.ones((number_of_samples, 1)),tc='d')
		G = matrix(-np.eye(number_of_samples),tc='d')
		h = matrix(np.zeros(number_of_samples),tc='d')
		A = matrix(Y.reshape(1, -1),tc='d')
		b = matrix(np.zeros(1),tc='d')
		solutions = solvers.qp(P,q,G,h,A,b)

		self.solutions=solutions
		return 


	def transform_under_constraints(self,C=None):	

		self.solutions=np.ravel(self.solutions['x'])
		if C :
			if C<0:
				print 'Error: Positive value needed' 
				return
			else:
				return np.array(map(lambda x: x if (x>=0 and x<self.margin) else 0,self.solutions))
		else:
			return np.array(map(lambda x: x if x>=0 else 0,self.solutions))

				
	def predict_model(self,X):

		if not type(X).__module__ == np.__name__:
			print "Error: X to predict must be numpy.ndarray .."
			sys.exit(0)

		output=np.array([])
		self.alph=self.transform_under_constraints()
		
		WX=np.zeros(X.shape[0])
		b0=[]

		for x in X:		
			if not self.kernel:

				for x_ii,y_ii,alph_ii in zip(X,self.Y,self.alph):
						WX+=y_ii*alph_ii*Linear(x_ii,x.T)
						
			if self.kernel=="rbf":
				for x_ii,y_ii,alph_ii in zip(X,self.Y,self.alph):
						WX+=y_ii*alph_ii*RBF(x_ii,x.T)

			if self.kernel=="tanh":
				for x_ii,y_ii,alph_ii in zip(X,self.Y,self.alph):
						WX+=y_ii*alph_ii*Hyperbolic_Tangent(x_ii,x.T)

			if self.kernel=="poly":
				for x_ii,y_ii,alph_ii in zip(X,self.Y,self.alph):
						WX+=y_ii*alph_ii*Polynomial(x_ii,x.T)

			for y_i in self.Y:
				b0.append([list(y_i-WX)])
			 		

		b=np.mean(np.array(b0))

		output=np.append(output,np.sign(WX+b))
		output=np.array(map(lambda x:self.dict[x],output))

		return output

def main(X,Y,X_predict,kernel=None):
	if kernel:
		model=SVM(X,Y,kernel,0.1)
		model.construct_model()
		print model.predict_model(X_predict)
	else:
		model=SVM(X,Y,0.1)
		model.construct_model()
		print model.predict_model(X_predict)


if __name__ == '__main__':

	X=np.array([[1,1,2],[0,3,1],[0,-2,1],[0,0,-1]])
	Y=np.array([1,-1,-1,1])
	X_predict=np.array([[-5,-3,-4],[5,10,29]])

	main(X,Y,X_predict,'tanh')
	

#print finetune_model(X,Y,,np.array([[8,9,8,9],[1,1,-3,2],[1,0,1,-1]]))



			




# print 'The primal objective is: ',sol['primal objective']



