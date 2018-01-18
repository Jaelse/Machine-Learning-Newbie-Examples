import matplotlib.pyplot as plt
import numpy
from mpl_toolkits.mplot3d import Axes3D

#3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

fig.subplots_adjust(top=0.85)
ax.set_title('Normal Equation')


#importing dataset
datain = numpy.loadtxt('weight.txt')

x0 = numpy.matrix(numpy.ones(100)).transpose()
X = numpy.concatenate((x0,datain[:,0:2]),1)
Y = datain[:,2:]

Xtrain = X[:-20,:]
Xtest = X[-20:,:]
Ytrain = Y[:-20,:]
Ytest = Y[-20:,:]

# Finding the theta
theta_opt = numpy.linalg.inv(Xtrain.transpose() * Xtrain) * Xtrain.transpose() * Ytrain

# plotting actual values and predicted value
ax.scatter(Xtest[:,1],Xtest[:,2], Xtest* theta_opt)
ax.scatter(Xtest[:,1],Xtest[:,2], Ytest,c='red')

#The Cost
cost_function = ((Xtest * theta_opt - Ytest).transpose() * (Xtest * theta_opt - Ytest))/len(Xtest)

ax.text2D(0.70, 0.95, "Predicted Value:BLUE\nReal Value:RED\nCost:"+str(cost_function[0,0]), transform=ax.transAxes)
ax.set_xlabel('Height', fontsize=15, rotation = 0)
ax.set_ylabel('Age', fontsize=15, rotation = 0)
ax.set_zlabel('Weight', fontsize=15, rotation = 0)

plt.show()
