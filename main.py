import modul
# BeratSoykuvvet
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

datas=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Makine Öğrenmesi\\Ödev\\maas.csv")
df_datas=pd.DataFrame(datas)
salary=df_datas.iloc[:,-1]
#P value process (backward elimination)
df_without_title=datas.drop(("unvan"),axis=1)
x=df_without_title.drop(("maas"),axis=1)#bagımsız 
x=x.iloc[:,[1,2,3]]
y=df_without_title.iloc[:,-1]#bagımlı
row=df_without_title.shape[0]#row sayısını verir
x=np.append(arr=np.ones((row,1)).astype(int),values=x,axis=1)
arr_l=np.array(x,dtype=float)
model=sm.OLS(y,x).fit()
df_datas=df_datas.iloc[:,[2,3,4]]
ModelSelection=input("1)Linear\n2)Polynomial\n3)Decision Tree\n4)Random Forest\n5)SVR\nPlease select the model:")
if ModelSelection=="1":
    title_level=int(input("please enter the title level:"))
    seniority=int(input("please enter the seniority:"))
    point=int(input("please enter the employee's point:"))
    df_emp=pd.DataFrame(data=[[title_level,seniority,point]],columns=["UnvanSeviyesi" , "Kidem" , "Puan"])
    modul.LinearRegModel(df_datas,salary,df_emp)
elif ModelSelection=="2":
    title_level=int(input("please enter the title level:"))
    seniority=int(input("please enter the seniority:"))
    point=int(input("please enter the employee's point:"))
    degree=int(input("Please enter the degree"))
    df_emp=pd.DataFrame(data=[[title_level,seniority,point]],columns=["UnvanSeviyesi" , "Kidem" , "Puan"])
    modul.PolynomialRegModel(df_datas,salary,degree,df_emp)
elif ModelSelection=="3":
    title_level=int(input("please enter the title level:"))
    seniority=int(input("please enter the seniority:"))
    point=int(input("please enter the employee's point:"))
    df_emp=pd.DataFrame(data=[[title_level,seniority,point]],columns=["UnvanSeviyesi" , "Kidem" , "Puan"])
    modul.DecisionTreeRegModel(df_datas,salary,df_emp)
elif ModelSelection=="4":
    title_level=int(input("please enter the title level:"))
    seniority=int(input("please enter the seniority:"))
    point=int(input("please enter the employee's point:"))
    df_emp=pd.DataFrame(data=[[title_level,seniority,point]],columns=["UnvanSeviyesi" , "Kidem" , "Puan"])
    modul.RandomForestRegModel(df_datas,salary,df_emp)   
elif ModelSelection=="5":
    title_level=int(input("please enter the title level:"))
    seniority=int(input("please enter the seniority:"))
    point=int(input("please enter the employee's point:"))
    kernel=input("please enter the kernel you want:")
    df_emp=pd.DataFrame(data=[[title_level,seniority,point]],columns=["UnvanSeviyesi" , "Kidem" , "Puan"])
    modul.SVRModel(df_datas,salary,kernel,df_emp)   
else:
    print("System down")
    
    
    
