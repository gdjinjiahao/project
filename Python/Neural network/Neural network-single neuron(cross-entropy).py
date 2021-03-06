import numpy as np
import matplotlib.pyplot as plt

#关于交叉熵，可以参考https://www.zhihu.com/question/41252833/answer/108777563
#还有小本本
#下边网址介绍了为什么均方误差会出现突变的情况（本质是sigmoid函数在原点导数极小，而交叉熵求导后可小区sigmoid导数）
#http://neuralnetworksanddeeplearning.com/chap3.html
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def update(w,b,y,n):
    #交叉熵在单神经元中化简后得到的梯度，y（即神经元输出）
    delta_w=-1.0*y*n
    delta_b=-1.0*y*n

    return w+delta_w,b+delta_b

w=10#0.6
b=10#0.9

epochs=300

x_data=[]
y_data=[]

for i in range(epochs):
    y=sigmoid(w+b)
    w,b=update(w,b,y,0.15)
    x_data.append(i)
    y_data.append(y)

plt.plot(x_data, y_data, linewidth=1, color="orange")
plt.title("matplotlib")
plt.xlabel("epochs")
plt.ylabel("output")
# 设置图例
plt.legend(["Y","Z"], loc="upper right")
#plt.grid(True)
plt.show()

#最后大致收敛至0.01