import matplotlib.pyplot as plt
import numpy
from mpl_toolkits.mplot3d import Axes3D

#3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

fig.subplots_adjust(top=0.85)
ax.set_title('Batch gradient')

# importing data
datain = numpy.loadtxt('weight.txt')

x0 = numpy.matrix(numpy.ones(100)).transpose()
X = numpy.concatenate((x0,datain[:,0:2]),1)
Y = datain[:,2:]

Xtrain = X[:-20,:]
Xtest = X[-20:,:]
Ytrain = Y[:-20,:]
Ytest = Y[-20:,:]

# initial theta values
theta_cur = numpy.matrix([-28, 0.5, 0.2]).transpose()

storing_residual = numpy.empty([10, 1])

for iter in range(0,10):
    # calculating the residual
    residual = (Xtrain * theta_cur - Ytrain).transpose() * (Xtrain * theta_cur - Ytrain)

    # recording
    storing_residual[iter,0] = residual

    # delta values
    delta_intercept = numpy.sum(Xtrain * theta_cur - Ytrain)
    delta_height = numpy.sum((Xtrain * theta_cur - Ytrain).transpose()*Xtrain[:,1])
    delta_age = numpy.sum((Xtrain * theta_cur - Ytrain).transpose()*Xtrain[:,2])

    # gradient
    gradient = numpy.matrix([delta_intercept, delta_height, delta_age]).transpose()

    # new theta value
    theta_cur = theta_cur - 0.00001/len(Xtrain) * gradient

cost_function = ((Xtest * theta_cur - Ytest).transpose() * (Xtest * theta_cur - Ytest))/len(Xtest)

# Plotting 3d graph for actual values and predicted values
ax.scatter(Xtest[:,1],Xtest[:,2], Xtest* theta_cur)
ax.scatter(Xtest[:,1],Xtest[:,2], Ytest,c='red')
ax.text2D(0.70, 0.95, "Predicted Value:BLUE\nReal Value:RED\nCost:"+str(cost_function[0,0]), transform=ax.transAxes)
ax.set_xlabel('Height', fontsize=15, rotation = 0)
ax.set_ylabel('Age', fontsize=15, rotation = 0)
ax.set_zlabel('Weight', fontsize=15, rotation = 0)


plt.show()

plt.title('epochs vs. J($\Theta$)')

plt.plot(range(0,10), storing_residual/len(Xtrain))
plt.scatter(range(0,10), storing_residual/len(Xtrain))
plt.xlabel('epochs')
plt.ylabel('Cost')
plt.show()
