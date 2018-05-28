import numpy as np
import matplotlib.pyplot as plt

'''
本程序和Neural network-single neuron(cross-entropy).py程序对应
只不过添加了一个神经元
但发现仅一层参数有明显变化，观察cost function发现交叉熵求导只能消除第一层的sigmoid倒数，但无法消除其他层，
故参数偏差较大时其他层梯度极小，几乎无法学习
直觉上和单层效果应该相近
但实验结果表明两层收敛效果是单层的10倍。。。。。。而且快速收敛所用代数为5左右，单层却需要50左右
why？？明明中间结果b和1只差了10的-9次方
'''

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def update(w1,w2,b1,b2,y,b,n):
    delta_w1=-1.0*y*w2*b*(1-b)*n
    delta_b1=delta_w1
    delta_w2=-1.0*y*b
    delta_b2=-1.0*y

    return w1+delta_w1,b1+delta_b1,w2+delta_w2,b2+delta_b2

w1,b1,w2,b2=1,0,1,0#10,10,10,10

epochs=300

x_data=[]
y_data=[]

for i in range(epochs):
    a=w1+b1
    b=sigmoid(a)
    t=w2*b+b2

    y=sigmoid(t)
    w1,b1,w2,b2=update(w1,w2,b1,b2,y,b,0.15)
    x_data.append(i)
    y_data.append(w1)

plt.plot(x_data, y_data, linewidth=1, color="orange")
plt.title("matplotlib")
plt.xlabel("epochs")
plt.ylabel("output")
# 设置图例
plt.legend(["Y","Z"], loc="upper right")
#plt.grid(True)
plt.show()

print(w1,b1,w2,b2,sigmoid(w1+b1))
#最后大致收敛至0.001
#print(sigmoid(10))
#0.9999999979388463
