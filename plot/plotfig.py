# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 18:24:14 2020
@author: chenj
"""
# 导入 pandas 和 matplotlib
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
# 读取文件
# =============================================================================
# 可能遇到的问题 路径分隔符 建议用“/”或“\\”  读取桌面文件时 用“\”可能会失败
# =============================================================================
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

data_source = pd.read_excel('C:/Users/14775/Desktop/Pagers/paper-1/图/曲线图.xlsx')
save_dir = 'C:/Users/14775/Desktop/Pagers/paper-1/图'
# 函数plot()尝试根据数字绘制出有意义的图形
# print(data_source['YOLOv71'])
mAP51 = data_source['YOLOv71']
mAP52 = data_source['B11']
mAP53 = data_source['B21']
mAP54 = data_source['B31']
mAP55 = data_source['B41']
plt.plot(mAP51, label='YOLOv7')
plt.plot(mAP52, label='B1')
plt.plot(mAP53, label='B2')
plt.plot(mAP54, label='B3')
pt = plt.plot(mAP55, label='B4')

fig = pt[0].figure
ax = pt[0].axes

plt.legend()  #显示上面的label
plt.xlabel('epoch')
plt.ylabel('mAP0.5')    #accuracy

# 局部放大
axins = zoomed_inset_axes(ax, 3, loc=10, borderpad=0.8)  # zoom = 6
axins.plot(mAP51, label='YOLOv7')
axins.plot(mAP52, label='B1')
axins.plot(mAP53, label='B2')
axins.plot(mAP54, label='B3')
axins.plot(mAP55, label='B4')

# 要放大的区域
x1, x2, y1, y2 = 250, 300, 0.47, 0.53
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

# 隐藏缩放坐标轴刻度线
axins.yaxis.set_visible(False)
axins.xaxis.set_visible(False)

plt.xticks(visible=False)
plt.yticks(visible=False)

# 放大区域要显示的位置
# connecting lines between the bbox and the inset axes area
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

plt.draw()
plt.savefig(Path(save_dir) / 'mAP0.5-epoch.png', dpi=200)
plt.show()

mAP1 = data_source['YOLOv7']
mAP2 = data_source['B1']
mAP3 = data_source['B2']
mAP4 = data_source['B3']
mAP5 = data_source['B4']
plt.plot(mAP1, label='YOLOv7')
plt.plot(mAP2, label='B1')
plt.plot(mAP3, label='B2')
plt.plot(mAP4, label='B3')
pt = plt.plot(mAP5, label='B4')

fig = pt[0].figure
ax = pt[0].axes

plt.legend()  #显示上面的label
plt.xlabel('epoch')
plt.ylabel('mAP')   #accuracy

axins = zoomed_inset_axes(ax, 3, loc=10, borderpad=0.8)  # zoom = 6
axins.plot(mAP1, label='YOLOv7')
axins.plot(mAP2, label='B1')
axins.plot(mAP3, label='B2')
axins.plot(mAP4, label='B3')
axins.plot(mAP5, label='B4')

# sub region of the original image
x1, x2, y1, y2 = 250, 300, 0.27, 0.305
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

# 隐藏缩放坐标轴刻度线
axins.yaxis.set_visible(False)
axins.xaxis.set_visible(False)

plt.xticks(visible=False)
plt.yticks(visible=False)

# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

plt.draw()
plt.savefig(Path(save_dir) / 'mAP-epoch.png', dpi=200)
plt.show()