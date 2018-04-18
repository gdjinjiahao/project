import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def update(w,b,y,n):
    delta_w=1.0*n*y*y*(y-1)
    delta_b=1.0*n*y*y*(y-1)

    return w+delta_w,b+delta_b
#去2时会有突变
w=0.5
b=0.5

epochs=300

#print(sigmoid(w+b))

x_data=[]
y_data=[]

for i in range(epochs):
    y=sigmoid(w+b)
    w,b=update(w,b,y,0.15)
    x_data.append(i)
    y_data.append(y)
    if i==epochs-1 :
        print(y)

print(w,b)

plt.plot(x_data, y_data, linewidth=1, color="orange")
plt.title("matplotlib")
plt.xlabel("epochs")
plt.ylabel("output")
# 设置图例
plt.legend(["Y","Z"], loc="upper right")
#plt.grid(True)
plt.show()
