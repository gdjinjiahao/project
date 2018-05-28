'''
中心极限定律和matplotlib绘图
'''
import numpy as np
import matplotlib.pyplot as plt

random_data=np.random.randint(1,7,10000)
'''
print(random_data.mean())
print(random_data.std())
'''

'''
sample1=[]
for i in range(0,10):
    sample1.append(random_data[int(np.random.random()*len(random_data))])

print(sample1)
'''


samples_mean=[]
samples_std=[]

for i in range(1000):
    sample=[]
    for j in range(50):
        sample.append(random_data[int(np.random.random()*len(random_data))])
    sample_np=np.array(sample)
    samples_mean.append(sample_np.mean())
    samples_std.append(sample_np.std())


print(samples_mean)

#samples_mean=[1,1,2,3,3,3,4,4,4,5,5,5,6,6,6]
fig=plt.figure()
ax=fig.add_subplot(111)
ax.hist(samples_mean,bins=20)
plt.show()

