'''
    Copyright (c) 2022 Idiap Research Institute, http://www.idiap.ch/
    Written by Suhan Shetty <suhan.shetty@idiap.ch>,
   
    This file is part of TTGO.

    TTGO is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License version 3 as
    published by the Free Software Foundation.

    TTGO is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with TTGO. If not, see <http://www.gnu.org/licenses/>.
'''


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator
import warnings
from matplotlib.colors import LogNorm
from matplotlib import ticker, cm
warnings.filterwarnings("ignore")

def plot_surf(x,y,cost,data=None,zlim=(0,1000),figsize=10, view_angle=(45,45),markersize=3):
   
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig.set_size_inches(figsize,figsize)

    Z = np.empty((len(x),len(y)))

    X, Y = np.meshgrid(x, y)
    XY = np.array([X.reshape(-1,),Y.reshape(-1,)]).T
    Z = cost(XY).reshape(X.shape[0],X.shape[1])

    cmap = sns.cm.rocket_r
    surf = ax.plot_surface(X, Y, Z,cmap=cmap,
                           linewidth=0, antialiased=False, zorder=0, alpha=1)
    # Customize the z axis.
    ax.set_zlim(zlim[0], zlim[1])
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.1f}')

    if not (data is None):
        data_z = 0
        if len(data.shape)==3:
            data_z = data[:,2] 
        ax.plot(data[:,0],data[:,1],data_z,'ob', markersize=markersize, zorder=10)
        
    ax.view_init(view_angle[0], view_angle[1])
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.4, aspect=5)
    
    return plt

def plot_contour(x,y,cost, data=None, contour_scale=100, figsize=10, markersize=3,log_norm=True):
    plt.style.use('seaborn-white')
    Z = np.empty((len(x),len(y)))
    X, Y = np.meshgrid(x, y)
    XY = np.array([X.reshape(-1,),Y.reshape(-1,)]).T
    Z = cost(XY).reshape(X.shape[0],X.shape[1])
    sns.set_style("white")
    cmap = 'binary_r'    
    if log_norm == True:
        levels = 10**(0.25*np.arange(-6,14))
        cs = plt.contour(X, Y, Z, contour_scale, cmap=cmap, shade=True,locator=ticker.LogLocator(),
            levels=levels, norm=LogNorm(), alpha=1);
    else:
        cs = plt.contour(X, Y, Z, contour_scale, cmap=cmap, shade=True,alpha=1);

    plt.colorbar(cs);
    if not (data is None):
        plt.plot(data[:,0],data[:,1],'ob', markersize=markersize)
    plt.rcParams["figure.figsize"] = (figsize, figsize)

    return plt
