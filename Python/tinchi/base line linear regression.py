import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

data=pd.read_csv("D:\\tianchi\\data\\user_balance_table.csv",header=0,encoding="utf8")

#现实头5条数据
#print(data.head())


x=[[i+1] for i in range(427)]




y1=[]
y2=[]


date=pd.date_range("20130701",periods=427)

ith=1

for i in date:
    tdata=data[data.report_date==i.year*10000+i.month*100+i.day]

    t1=tdata["total_purchase_amt"].sum()
    t2=tdata["total_redeem_amt"].sum()

    y1.append([t1])
    y2.append([t2])

    print(ith*1.0/427)
    ith=ith+1

reg1=linear_model.LinearRegression()
reg2=linear_model.LinearRegression()

reg1.fit(x,y1)
reg2.fit(x,y2)

print("fit ok")

px=[[i+428] for i in range(30)]

py1=reg1.predict(px)
py2=reg2.predict(px)

for i in range(30):
    py1[i]=py1[i]*100
    py2[i]=py2[i]*100

report_date=[]
for i in range(30):
    report_date.append(20140901+i)

columns=["report_date","purchase","redeem"]
save=pd.DataFrame({"report_date":report_date,"purchase":py1.flat[:],"redeem":py2.flat[:]},dtype=np.int64)
save.to_csv("D:\\tc_comp_predict_table.csv",index=False,header=False,columns=columns)

print("done!")