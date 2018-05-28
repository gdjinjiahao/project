'''
该模型的损失函数为min sum{(wx-y)^2}，x和y均为观测值
即求最小平方和
适合数据独立的情况
若相互联系并且有各观测项有近似的线性关系，则一旦出现少量无差点
会对方差造成比较大的影响，从而影响参数进化，导致误差较大
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error,r2_score

# Load the diabetes dataset
diabetes = datasets.load_diabetes()


# Use only one feature
#diabetes_X = diabetes.data[:, np.newaxis, 2]
'''
#这是一个正常的线性回归
# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]
'''

#这个是故意在相关性强的数据集中设置错误点的例子
diabetes_X_train = [[i] for i in range(20)]
diabetes_X_test = [[i+22] for i in range(10)]

diabetes_y_train = [i for i in range(20)]
diabetes_y_test = [i+22 for i in range(10)]

diabetes_X_train.append([21])
diabetes_y_train.append(30)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='red')
plt.scatter(diabetes_X_train, diabetes_y_train,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

#reg类中coef_存储权值，intercept_存储偏置
#print("Weights are :",reg.coef_)
#print("Bias is :",reg.intercept_)