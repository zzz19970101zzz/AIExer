# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from sklearn.linear_model import LinearRegression
import  matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression

# x = [[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10]]
# y = [[1,2,3,4,5,6,7,8,9,10],[7,9,11,13,15,17,19,21,23,25]]
# lr_model = LinearRegression()
# lr_model.fit(x,y)
# # a=lr_model.coef_
# b=lr_model.intercept_
# print(a)
# printrint(b)
# print(lr_model.predict(10))
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    x = 2 * np.random.rand(100,1)
    y = 4 + 3 * x + np.random.randn(100,1)
    x_b = np.c_[np.ones((100,1)),x]
    theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)

    x_new  = np.array([[0],[2]])
    x_new_b = np.c_[np.ones((2,1)),x_new]
    y_predict = x_new_b.dot(theta_best)
    plt.plot(x_new,y_predict,'r-')
    plt.plot(x,y,'b.')
    plt.axis([0,2,0,15])
    plt.xlabel('x')
    plt.ylabel('y_predict')
    # plt.show()
    print()

    # lin_reg =  LinearRegression()
    # lin_reg.fit(x,y)
    # print(lin_reg.intercept_)
    # print(lin_reg.coef_)

    # eta = 0.5
    # n_iterations = 1000
    # m =100
    # theta = np.random.randn(2,1)
    # for iterations in range(n_iterations):
    #     gradients = 2/m * x_b.T.dot(x_b.dot(theta) - y)
    #     theta = theta - eta * gradients
    # print(theta)

    sgd_reg = SGDRegressor(max_iter=10000,tol=1e-3,penalty=None,eta0=0.1)
    sgd_reg.fit(x,y.ravel())
    print(sgd_reg.intercept_,sgd_reg.coef_)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
