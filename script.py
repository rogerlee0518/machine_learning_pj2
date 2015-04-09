import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
import matplotlib.pyplot as plt
import pickle
'''
def ldaLearn(X,y):
        # Inputs
        # X - a N x d matrix with each row corresponding to a training example
        # y - a N x 1 column vector indicating the labels for each training example
        #
        # Outputs
        # means - A d x k matrix containing learnt means for each of the k classes
        # covmat - A single d x d learnt covariance matrix 

        # IMPLEMENT THIS METHOD

        return means,covmat

def qdaLearn(X,y):
        # Inputs
        # X - a N x d matrix with each row corresponding to a training example
        # y - a N x 1 column vector indicating the labels for each training example
        #
        # Outputs
        # means - A d x k matrix containing learnt means for each of the k classes
        # covmats - A list of k d x d learnt covariance matrices for each of the k classes

        # IMPLEMENT THIS METHOD

        return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
        # Inputs
        # means, covmat - parameters of the LDA model
        # Xtest - a N x d matrix with each row corresponding to a test example
        # ytest - a N x 1 column vector indicating the labels for each test example
        # Outputs
        # acc - A scalar accuracy value

        # IMPLEMENT THIS METHOD
        return acc

def qdaTest(means,covmats,Xtest,ytest):
        # Inputs
        # means, covmats - parameters of the QDA model
        # Xtest - a N x d matrix with each row corresponding to a test example
        # ytest - a N x 1 column vector indicating the labels for each test example
        # Outputs
        # acc - A scalar accuracy value

        # IMPLEMENT THIS METHOD
        return acc
'''
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
        xw = np.dot(X,w)	# X*w
        # 1/2N * (y-xw)^T * (y-xw)
        err_part1 = (np.sum(np.dot(np.transpose(y-xw),(y-xw))))/(2*N)
        # 1/2 * lambd * w^T * w
        err_part2 = (lambd*(np.dot(np.transpose(w),w)))/2
        # J(w)
        error = err_part1 + err_part2
        xtx = np.dot(np.transpose(X),X)
        ytx = np.dot(np.transpose(y),X)
        # 1/N * [w^T * (x^T * x) - y^T * x]
        err_grad1 = (np.dot(np.transpose(w),xtx)-ytx) / N
        # lambd * w
        err_grad2 = lambd * w
        error_grad = err_grad1 + err_grad2
        print ('error')
        print (error)
        print ('error gradient')
        print (error_grad)
        error_grad = np.array(error_grad).reshape(-1)
        return error, error_grad;
'''
        #CALVIN CODE
        N = X.shape[0];
        A = np.dot(y.T, y);
        B = np.dot(2, np.dot(y.T, np.dot(X, w)));
        C = np.dot(w.T, np.dot(X.T, np.dot(X, w)));
        D = np.dot(np.dot(1/2, N), A+B+C);
        E = np.dot(1/2, np.dot(lambd, np.dot(w.T, w)));
        error = D + E;

        F = np.dot(w.T, np.dot(X.T, X));
        G = np.dot(y.T, X);
        H = np.dot(1/N, F+G);
        error_grad = H + np.dot(lambd, w);
        return error, error_grad

        #RANDOM CODE
        N = X.shape[0];
        A = np.dot(X, w);
        B = pow(y-A, 2);
        C = np.dot(lambd, np.dot(w.T, w));
        error = np.dot(1/N, np.sum(B)+C);

        D = np.dot(X, w);
        E = np.dot((2/N), np.dot(D, X));
        F = np.dot(lambd, w.T);
        error_grad = (E+F).T;

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
'''
# Main script

# Problem 1
# load the sample data                                                                 
X,y,Xtest,ytest = pickle.load(open('/home/sean/code/school/cse474/machine_learning_pj2/sample.pickle','rb'))            
'''
# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# Problem 2
'''

#X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)
'''
w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

# Problem 3
k = 21
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeERegression(X_i,y,lambd)
        rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
        i = i + 1
plt.plot(lambdas,rmses3)
'''

# Problem 4
k = 21
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

'''
# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
        Xdtest = mapNonLinear(Xtest[:,2],p)
        w_d1 = learnRidgeRegression(Xd,y,0)
        rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
        w_d2 = learnRidgeRegression(Xd,y,lamda_opt)
        rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
'''
