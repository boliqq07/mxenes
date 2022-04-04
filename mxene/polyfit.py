# import numpy as np
# import pandas as pd
# import scipy
# from numpy.linalg import norm
# from scipy.interpolate import interp2d, RectBivariateSpline, SmoothBivariateSpline, BivariateSpline, griddata, \
#     LinearNDInterpolator, CloughTocher2DInterpolator
#
# frac = pd.read_csv("../test/mx.cart_coords.csv").values
# frac = frac[np.where(frac[:,-1]>12.7)]
#
#
# tx,ty,tz =frac[:,0],frac[:,1],frac[:,2]
#
# import matplotlib.pyplot as plt
#
# # xnew = np.arange(-12,23,0.2)
# # ynew = np.arange(0,21,0.2)
# #
# #
# # Xnew, Ynew = np.meshgrid(xnew, ynew)
# # Znew = f(xnew, ynew).T
#
# # f = CloughTocher2DInterpolator(frac[:,:2], z,tol=0.1)
# # Znew = f(Xnew, Ynew)
# # f = interp2d(x, y, z,kind="linear")
# # Znew = f(xnew, ynew)
#
#
# # plt.contourf(xnew, ynew, Znew,)
# # plt.imshow(Znew,cmap=plt.get_cmap('jet'))
# # plt.show()
#
# # from mpl_toolkits.mplot3d import Axes3D
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # ax.plot_trisurf(x,y,z)
# # # 显示图表
# # plt.show()
# # x=np.arange(0,4,0.1)
# # y = np.exp(-x+2)
#
# x = np.arange(-12,23,0.2)
# y = np.arange(0,21,0.2)
#
# x_mesh, y_mesh = np.meshgrid(x,y)
# x_mesh = np.ravel(x_mesh)
# y_mesh = np.ravel(y_mesh)
# data = np.hstack((x_mesh.reshape(-1,1),y_mesh.reshape(-1,1)))
#
# # t_data = np.hstack((tx.reshape(-1,1), ty.reshape(-1,1)))
# # dis = np.sum((t_data - data[0]) * 2,axis=1) ** 0.5
# # w = np.exp(-dis*2 + 2)
# # w=w/np.sum(w)
# # z=np.sum(w*tz)
#
# t_data = np.hstack((tx.reshape(-1,1), ty.reshape(-1,1)))
# t_data_ = t_data[:,np.newaxis,:]
# t_data_ = np.repeat(t_data_,data.shape[0],axis=1)
# data_ = data[np.newaxis,:,:]
# data_ = np.repeat(data_,t_data.shape[0],axis=0)
# dis = np.sum((t_data_- data_) ** 2, axis=-1)**0.5
# w = np.exp(-dis*2)
# w = w/np.sum(w,axis=0)
# z=np.sum(w*np.repeat(tz.reshape(-1,1),w.shape[1],axis=1),axis=0)
# z = z.reshape((y.shape[0],x.shape[0]))
#
# plt.imshow(z)
# plt.show()
