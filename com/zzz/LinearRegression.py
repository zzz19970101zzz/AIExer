import numpy as np
from utils.features import prepare_for_training

class LinearRegression:
    def __init__(self,data,labels,polynomial_degree = 0,sinusoid_degree = 0,normalize_data=True):
        [data_processed,
         features_mean,
         features_deviation] = prepare_for_training(data,labels,polynomial_degree = 0,sinusoid_degree = 0,normalize_data=True)
        self.data = data_processed
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data
        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features,1))

    def train(self,alpha,num_iterations = 500):
        cost_histry = self.gradient_descent(alpha,num_iterations)
        return self.theta,cost_histry

    def gradient_descent(self,alpha,num_iterations):
        cost_history = []
        for _ in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data,self.labels))
        return cost_history

    def gradient_step(self,alpha):
        num_examples = self.data.shape[0]
        prediction = LinearRegression.hypothesis(self.data,self.theta)
        delta = prediction - self.labels
        theta = self.theta
        theta = theta - alpha * (1 / num_examples) * (np.dot(delta.T,self.data)) . T
        self.theta = theta

    def cost_function(self,data,labels):
        num_examples = data.shape[0]
        delta = LinearRegression.hypothesis(self.data,self.theta) - labels
        cost = (1/2)*np.dot(delta.T,delta)
        return cost[0][0]
    @staticmethod
    def hypothesis(data,theta):
        predictions = np.dot(data ,theta)
        return predictions

    def get_cost(self,data,labels):
        data_processed = prepare_for_training(data,self.polynomial_degree,
         self.sinusoid_degree,self.normalize_data)[0]

        return self.cost_function(data_processed,labels)

    def predict(self,data):
        data_processed = prepare_for_training(data, self.polynomial_degree,
                                              self.sinusoid_degree, self.normalize_data)[0]
        predictions = LinearRegression.hypothesis(data_processed,self.theta)