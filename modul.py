# BeratSoykuvvet
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



#Linear Regression
def LinearRegModel(independent,dependat,employee):
    Linear_reg=LinearRegression()
    Linear_reg.fit(independent,dependat)
    predict_LR=Linear_reg.predict(employee)
    print("Salary for the employee:",predict_LR)


#polynomial regression
def PolynomialRegModel(independent,dependat,degree,employee):
    poly_reg=PolynomialFeatures(degree=degree)
    poly_reg_train=poly_reg.fit_transform(independent)
    poly_reg_test=poly_reg.fit_transform(employee)
    Linear_reg2=LinearRegression()
    Linear_reg2.fit(poly_reg_train,dependat)
    predict_poly_reg=Linear_reg2.predict(poly_reg_test)
    print("Salary for the employee:",predict_poly_reg)

#Decision tree regression
def DecisionTreeRegModel(independent,dependat,employee):
    DT_reg=DecisionTreeRegressor()
    DT_reg.fit(independent,dependat)
    predict_DT=DT_reg.predict(employee)
    print("Salary for the employee:",predict_DT)

#Random Forest regression
def RandomForestRegModel(independent,dependat,employee):
    RF_reg=RandomForestRegressor()
    y_train_np=np.array(dependat)
    RF_reg.fit(independent,y_train_np.ravel())
    predict_RF=RF_reg.predict(employee)
    print("Salary for the employee:",predict_RF)

#Support Vector Regression
def SVRModel(independent,dependat,kernel,employee):
    sc_x=StandardScaler()
    sc_y=StandardScaler()
    x_train_scaled=sc_x.fit_transform(independent)
    y_train=dependat.values.reshape(-1,1)
    y_train_scaled=sc_y.fit_transform(y_train).ravel()
    x_test_scaled=sc_x.transform(employee)
    y_test=dependat.values.reshape(-1,1)
    y_test_scaled=sc_y.transform(y_test).ravel()
    svr_reg=SVR(kernel=f"{kernel}")
    svr_reg.fit(x_train_scaled,y_train_scaled)
    predict_SVR=svr_reg.predict(x_test_scaled)
    predict_SVR_inversed = sc_y.inverse_transform(predict_SVR.reshape(-1, 1))
    print("Salary for the employee:",predict_SVR_inversed)


