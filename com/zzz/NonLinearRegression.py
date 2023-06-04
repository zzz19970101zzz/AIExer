import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from LinearRegression import LinearRegression

data = pd.read_csv('data/non-linear-regression-x-y.csv')
x = data['x'].values.reshape(data.shape[0],1)
y = data['y'].values.reshape(data.shape[0],1)
data.head(10)
plt.plot(x,y)
plt.show()
num_iterations = 50000
learning_rate = 0.02
polynomial_degree = 15
normalize_data = True
sinusoid_degree = 15

linear_regression = LinearRegression(x,y,polynomial_degree,sinusoid_degree,normalize_data)
(theta,cost_history) = linear_regression.train(learning_rate,num_iterations)
print("开始损失",cost_history[0])
print("训练后损失",cost_history[-1])

plt.plot(range(num_iterations),cost_history)
plt.ylabel('cost')
plt.xlabel('Iter')
plt.title('GD')
plt.show()

predictions_num = 100
# x_predictions = np.linspace(x_train.min(),x_train.max(),predictions_num).reshape(predictions_num,1)
y_predictions = linear_regression.predict(x)

plt.scatter(x,y,label = 'orign data')
 # plt.scatter(x_test,y_test,label = 'test_data')
plt.plot(x,y_predictions,'r',label = 'Prediction')
plt.xlabel('x')
plt.ylabel('y_predictions')
plt.title('Happy')
plt.legend()
plt.show()