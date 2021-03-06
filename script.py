import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
import matplotlib.pyplot as plt
import pickle

def ldaLearn(X,y):
	# Inputs
	# X - a N x d matrix with each row corresponding to a training example
	# y - a N x 1 column vector indicating the labels for each training example
	#
	# Outputs
	# means - A d x k matrix containing learnt means for each of the k classes
	# covmat - A single d x d learnt covariance matrix 

	# IMPLEMENT THIS METHOD
	N = X.shape[0]
	d = X.shape[1]
	classesnum=0
	classesarray=np.zeros(1)
	for i in range(0, y.shape[0]):
		if y[i] not in classesarray:
			classesnum = classesnum+1
			classesarray=np.append(classesarray,y[i])
	if 0 not in y:
		classesarray=np.delete(classesarray,0)
	else:
		classesnum=classesnum+1
	meanarray=np.zeros((classesarray.shape[0],d))
	classesarray=np.sort(classesarray)
	for i in range(0,classesarray.shape[0]):
		tempcount=0
		for j in range(0,y.shape[0]):
			if classesarray[i]==y[j]:
				tempcount=tempcount+1
				for k in range(0,X.shape[1]):
					meanarray[i][k]=meanarray[i][k]+X[j][k]
		for p in range(0,X.shape[1]):
			meanarray[i][p]=meanarray[i][p]/tempcount

	# initialization
	means=np.zeros((d,classesarray.shape[0]))
	for i in range(0,meanarray.shape[0]):
		for j in range(0,meanarray.shape[1]):
			means[j][i]=meanarray[i][j]

	return means, np.cov(X.T)
#return means,covmat

def qdaLearn(X,y):
	# Inputs
	# X - a N x d matrix with each row corresponding to a training example
	# y - a N x 1 column vector indicating the labels for each training example
	#
	# Outputs
	# means - A d x k matrix containing learnt means for each of the k classes
	# covmats - A list of k d x d learnt covariance matrices for each of the k classes
	N = X.shape[0]
	d = X.shape[1]
	classesnum=0
	classesarray=np.zeros(1)
	for i in range(0, y.shape[0]):
		if y[i] not in classesarray:
			classesnum = classesnum+1
			classesarray=np.append(classesarray,y[i])
	if 0 not in y:
		classesarray=np.delete(classesarray,0)
	else:
		classesnum=classesnum+1
	meanarray=np.zeros((classesarray.shape[0],d))
	classesarray=np.sort(classesarray)
	for i in range(0,classesarray.shape[0]):
		tempcount=0
		for j in range(0,y.shape[0]):
			if classesarray[i]==y[j]:
				tempcount=tempcount+1
				for k in range(0,X.shape[1]):
					meanarray[i][k]=meanarray[i][k]+X[j][k]
		for p in range(0,X.shape[1]):
			meanarray[i][p]=meanarray[i][p]/tempcount

	# initialization
	means=np.zeros((d,classesarray.shape[0]))
	for i in range(0,meanarray.shape[0]):
		for j in range(0,meanarray.shape[1]):
			means[j][i]=meanarray[i][j]

	covmats=[np.zeros((d,d))]*classesnum
	for i in range(0,classesarray.shape[0]):
		count=1
		for j in range(0,y.shape[0]):
			if classesarray[i]==y[j] and count==0:
				temp=np.vstack((temp,X[j]))
			elif classesarray[i]==y[j] and count==1:
				temp=np.array(X[j])
				count=0
		covmats[i]=np.cov(temp.T)
	return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
	# Inputs
	# means, covmat - parameters of the LDA model
	# Xtest - a N x d matrix with each row corresponding to a test example
	# ytest - a N x 1 column vector indicating the labels for each test example
	# Outputs
	# acc - A scalar accuracy value

	# IMPLEMENT THIS METHOD
	covmatint = np.linalg.inv(covmat)
	covmatdet = np.linalg.det(covmat)
	meansT=means.T
	pi=np.pi
	temp=np.zeros(means.shape[1])
	count=0
	for i in Xtest:
		tempslice=np.array([])
		if count==1:
			for j in meansT:
				tempslice=np.hstack((tempslice, ((0.5)*covmatdet*np.sqrt(2*pi))*np.exp((-0.5)*(np.dot((i-j),(i-j).T/(covmatdet**2))))))
			temp=np.vstack((temp,tempslice.T))
			#rint(tempslice)
		elif count==0:
			for j in meansT:
				tempslice=np.hstack((tempslice, ((0.5)*covmatdet*np.sqrt(2*pi))*np.exp((-0.5)*(np.dot((i-j),(i-j).T/(covmatdet**2))))))
			temp=tempslice
			count=1
	prob = np.argmax(temp, axis=1)
	prob = np.add(prob,1)
	acc = 0.0
	#If our value is true then our predicition matches the ytest
	for i in range(0,ytest.shape[0]):
		if (ytest[i][0] == prob[i]):
			acc = acc + 1
	acc = acc/temp.shape[0]
	# IMPLEMENT THIS METHOD
	return acc

def qdaTest(means,covmats,Xtest,ytest):
	# Inputs
	# means, covmat - parameters of the LDA model
	# Xtest - a N x d matrix with each row corresponding to a test example
	# ytest - a N x 1 column vector indicating the labels for each test example
	# Outputs
	# acc - A scalar accuracy value

	# IMPLEMENT THIS METHOD
	covmatint = np.linalg.inv(covmat)
	covmatdet = np.linalg.det(covmat)
	meansT=means.T
	pi=np.pi
	temp=np.zeros(means.shape[1])
	count=0
	for i in Xtest:
		tempslice=np.array([])
		if count==1:
			for j in meansT:
				tempslice=np.hstack((tempslice, ((0.5)*covmatdet*np.sqrt(2*pi))*np.exp((-0.5)*(np.dot((i-j),(i-j).T/(covmatdet**2))))))
			temp=np.vstack((temp,tempslice.T))
			#print(tempslice)
		elif count==0:
			for j in meansT:
				tempslice=np.hstack((tempslice, ((0.5)*covmatdet*np.sqrt(2*pi))*np.exp((-0.5)*(np.dot((i-j),(i-j).T/(covmatdet**2))))))
			temp=tempslice
			count=1
	prob = np.argmax(temp, axis=1)
	prob = np.add(prob,1)
	acc = 0.0
	#If our value is true then our predicition matches the ytest
	for i in range(0,ytest.shape[0]):
		if (ytest[i][0] == prob[i]):
			acc = acc + 1
	acc = acc/temp.shape[0]
	# IMPLEMENT THIS METHOD
	return acc

def learnOLERegression(X,y):
	# Inputs:                                                         
	# X = N x d 
	# y = N x 1                                                               
	# Output: 
	# w = d x 1                                                                
	# IMPLEMENT THIS METHOD 
	# w = (X.T*X)^-1 * X.T * y (lec.22)                                    
	A = np.linalg.inv(np.dot(X.T, X));
	B = np.dot(A, X.T);
	w = np.dot(B, y);
	return w;

def learnRidgeERegression(X,y,lambd):
	# Inputs:
	# X = N x d                                                               
	# y = N x 1 
	# lambd = ridge parameter (scalar)
	# Output:                                                                  
	# w = d x 1                                                                
	# IMPLEMENT THIS METHOD
	# w = (((X'X)+N*lambda*I)^-1)X'y

	N = X.shape[0];
	I = np.identity(X.shape[1]);
	C = np.dot(N, np.dot(lambd, I));
	E = np.linalg.inv(np.dot(X.T, X) + C);
	return np.dot(E, np.dot(X.T, y));

def testOLERegression(w,Xtest,ytest):
	# Inputs:
	# w = d x 1
	# Xtest = N x d
	# ytest = X x 1
	# Output:
	# rmse
	# IMPLEMENT THIS METHOD
	# J(w) = sqrt(sum((y-X*w)^2))*(1/N)
	A = ytest - np.dot(Xtest, w);       #Difference between y and x*w
	B = np.square(A);                   #Power of 2
	C = np.sum(B);                      #Add the summation to N
	D = np.sqrt(C);                     #Square root the function
	rmse = D/Xtest.shape[0];            #Divide result by N
	return rmse

def regressionObjVal(w, X, y, lambd):

	# compute squared error (scalar) and gradient of squared error with respect
	# to w (vector) for the given data X and y and the regularization parameter
	# lambda                                                                  
	# IMPLEMENT THIS METHOD  

	#SEAN CODE                                  
	N = X.shape[0]		# N
	y = y.reshape(242,)   
	xw = np.dot(X,w)	# X*w
	# 1/2N * (y-xw)^T * (y-xw)
	err_part1 = (np.sum(np.dot(np.transpose(y-xw),(y-xw))))*(1.0/(2*N))
	# 1/2 * lambd * w^T * w
	err_part2 = (lambd*(np.dot(np.transpose(w),w)))*(1.0/2.0)
	# J(w)
	error = err_part1 + err_part2
	xtx = np.dot(np.transpose(X),X)
	ytx = np.dot(np.transpose(y),X)
	# 1/N * [w^T * (x^T * x) - y^T * x]
	err_grad1 = (np.dot(np.transpose(w),xtx)-ytx)*(1.0/N)
	# lambd * w
	err_grad2 = lambd * w
	error_grad = err_grad1 + err_grad2
	#error_grad = np.linalg.inv(error_grad)
	error_grad = error_grad.flatten()
	return error, error_grad


def mapNonLinear(x,p):
	# Inputs:                                                                  
	# x - a single column vector (N x 1)                                       
	# p - integer (>= 0)                                                       
	# Outputs:                                                                 
	# Xd - (N x (d+1))                                                         
	# IMPLEMENT THIS METHOD
	A = np.shape(x)[0]
	Xd = np.zeros((A,p+1))
	for i in range(Xd.shape[1]):
		for k in range(x.shape[0]):
			Xd[k][i] = pow(x[k],i)
	return Xd

# Main script

# Problem 1
# load the sample data                                                                 
X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))            
# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# Problem 2

X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
	w_l = learnRidgeERegression(X_i,y,lambd)
	rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
	i = i + 1
plt.plot(lambdas,rmses3)
plt.show()

# Problem 4
k = 101
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.
w_init = np.zeros((X_i.shape[1],1))
for lambd in lambdas:
	args = (X_i, y, lambd)
	w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
	w_l_1 = np.zeros((X_i.shape[1],1))
	for j in range(len(w_l.x)):
		w_l_1[j] = w_l.x[j]
	rmses4[i] = testOLERegression(w_l_1,Xtest_i,ytest)
	i = i + 1
plt.plot(lambdas,rmses4)
plt.show()


# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses3)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
	Xd = mapNonLinear(X[:,2],p)
	Xdtest = mapNonLinear(Xtest[:,2],p)
	w_d1 = learnRidgeERegression(Xd,y,0)
	rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
	w_d2 = learnRidgeERegression(Xd,y,lambda_opt)
	rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
plt.show()
