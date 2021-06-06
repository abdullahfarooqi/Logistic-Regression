import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import fmin_tnc

#Opening The File
file1 = open("ex2data1.txt", "r+")
#Getting and Parsing the Data
training_data= pd.read_csv("ex2data1.txt")
#Appropriately Naming the Columns
training_data.columns =[
    "Score1",
    "Score2",
    "Label"
]
#Retreiving rows from iloc method
X=training_data.iloc[:,:-1]   # X= feature values, all the colums except the last column
Y=training_data.iloc[:,-1]     # Y= Actual Lable, Last Column of the data frame

# Filtering out the students that got admitted or not
admitted=training_data.loc[Y==1]
not_admitted=training_data[Y==0]

#plotting points as a scatter plot
plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1],marker='+',color='black',label='Admitted')
plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], color='yellow',edgecolors='k',s=20,label='Not Admitted')
# x-axis label
plt.xlabel('Exam1 Score')
# y-axis label
plt.ylabel('Exam2 Score')
# plot title
plt.title('My scatter plot!')
# showing legend
plt.legend()
# function to show the plot
plt.show()

# Data for our model
(m,n)= X.shape #Shape function returns the no. of elements in each dimension.
X = np.hstack((np.ones((m,1)), X))
Y= Y[:,np.newaxis]
theta= np.zeros((n+1,1))

# Function used to map any real value between 0 and 1
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Computing the weighted sum of inputs i.e theta^T.x
def net_input(theta,x):
    return np.dot(x, theta)

# Calculating the probability after passing through sigmoid i.e h(x)=g(theta^T.x)
def probability(theta,x):
    return sigmoid(net_input(theta, x))

# Calculating the cost for all training examples
def cost(theta,x,y):
    total_cost = -(1 / m) * np.sum(y * np.log(probability(theta, x)) + (1 - y) * np.log(1 - probability(theta, x)))
    return total_cost

# Calculating the gradient(slope) of the cost function at the point theta
def gradient(theta,x,y):
    return (1 / m) * np.dot(x.T, sigmoid(net_input(theta, x)) - y)

J = cost(theta, X, Y)
# Now,finding the optimal theta's for our model
#fmin_tnc is used to compute the minimum for any function. It takes arguments as
#func: the function to minimize
#x0: initial values for the parameters that we want to find
#fprime: gradient for the function defined by ‘func’
#args: arguments that needs to be passed to the functions.
opt_theta=fmin_tnc(func=cost, x0=theta,fprime=gradient,args=(X, Y.flatten()))
#The function is a tuple whose first element contains the optimal values of theta

Optimal_Theta= opt_theta[0]
J = cost(Optimal_Theta[:,np.newaxis], X, Y)

#Ploting the Decision Boundary
x_values = [np.min(X[:, 1] - 2), np.max(X[:, 2] + 5)]
y_values = - (Optimal_Theta[0] + np.dot(Optimal_Theta[1], x_values)) / Optimal_Theta[2]
X=training_data.iloc[:,:-1]   # X= feature values, all the colums except the last column
Y=training_data.iloc[:,-1]     # Y= Actual Lable, Last Column of the data frame
plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1],marker='+',color='black',label='Admitted')
plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], color='yellow',edgecolors='k',s=20,label='Not Admitted')
plt.plot(x_values, y_values, label='Decision Boundary')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.legend()
plt.show()

#Analyzing the accuracy of the model
def predict(X):
    theta = Optimal_Theta[:, np.newaxis]
    return probability(theta, X)

def accuracy(X, actual_classes, probab_threshold):
    predicted_classes = (predict(X) >= probab_threshold).astype(int)
    predicted_classes = predicted_classes.flatten()
    accuracy = np.mean(predicted_classes == actual_classes)
    return accuracy * 100

acc=accuracy(X, Y.values.flatten(),0.5)
print("Accuracy : ",acc)




