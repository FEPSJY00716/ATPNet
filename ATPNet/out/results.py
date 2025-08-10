import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

plt.rcParams['figure.figsize'] = (6.5, 5)  # 设置figure_size尺寸
# plt.rcParams['image.interpolation'] = 'nearest'  # 设置 interpolation style
# plt.rcParams['image.cmap'] = 'gray'  # 设置 颜色 style
# plt.rcParams['savefig.dpi'] = 300  # 图片像素
# plt.rcParams['figure.dpi'] = 300  # 分辨率
font = {'family': 'sans',
        'sans-serif': 'Times New Roman',
        'weight': 'normal',
        'size': 16}
plt.rc('font', **font)  # pass in the font dict as kwargs

data = pd.read_csv("results.csv", index_col=0)
print(data.head())
x = data['true'].values.reshape(-1, 1)
y = data['pred'].values
print(x.shape, y.shape)


linear = LinearRegression()
linear.fit(x, y)

info = "y = {:.2f} x + {:.2f}\n$R^2$ = {:.4f}".format(
    linear.coef_.item(), linear.intercept_.item(), linear.score(x, y))

plt.text(20, 10, info)
# plt.plot(x, linear.coef_ * x + linear.intercept_)
# plt.scatter(x, y, c="r")
# data.columns = ['预测值', '真实值']
# print(data.columns)
sns.regplot(x='true', y='pred', data=data)
# sns.lmplot(x='true', y='pred', data=data)
plt.savefig("regres.svg")
plt.show()