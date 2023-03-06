
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


import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def plot_chain(joint_loc, link_lengths, x_obst=[], r_obst=[], x_target=[], rect_patch=[], 
    batch=False, skip_frame=10, title=None, save_as=None,figsize=3,
    color_intensity=0.9, motion=False, alpha=0.5, contrast=0.4, idx_highlight=[], lw=7, task='ik'):

    fig = plt.figure(edgecolor=[0.1,0.1,0.1])

    fig.set_size_inches(figsize, figsize)
    sns.set_theme()

    sns.set_context("paper")


    # fig.patch.set_facecolor('white')
    # fig.patch.set_alpha(0.9)
    xmax_ = 1.1*np.sum(link_lengths)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(-xmax_, xmax_), ylim=(-xmax_, xmax_))

    for i,x in enumerate(x_obst):
        circ = plt.Circle(x,r_obst[i],color='grey',alpha=0.5)
        ax.add_patch(circ)
    for i, x_ in enumerate(rect_patch):
        rect = plt.Rectangle(rect_patch[i][0:2],rect_patch[i][2],rect_patch[i][3], color='c',alpha=0.5)
        ax.add_patch(rect)
    color_ = ['g','r']
    for i, x_ in enumerate(x_target):
        ax.plot(x_[0],x_[1],'or', markersize=10)

    

    if batch is False:
        x = joint_loc[:,0]
        y = joint_loc[:,1]
        k_ = 1*color_intensity
        color_ = [k_,k_,k_]
        plt.plot(x, y, 'o-',zorder=0, marker='o',color=color_,lw=lw,mfc='w',
                    solid_capstyle='round')
    else:

        T = joint_loc.shape[0]

        if task=='via':
            ax.legend(["target","obstacle"])
            idx = np.arange(0,int(T/2), skip_frame)
            k_ = np.linspace(0.3,0.7,len(idx))[::-1]
            k_[0]=1
            for count,i in enumerate(idx):
                # color_ = np.where(motion, 1-k_[count], contrast)
                x = joint_loc[i,:,0]
                y = joint_loc[i,:,1]
                plt.plot(x, y, 'o-',zorder=0.9,marker='o',color='g',lw=lw,mfc='w',
                            solid_capstyle='round', alpha= k_[count])
                plt.plot(joint_loc[i,-1,0],joint_loc[i,-1,1],'oy', markersize=3)

            idx = idx = np.arange(int(T/2),T, skip_frame)
            k_ = np.linspace(0.2,0.5,len(idx))
            k_[-1] = 1
            for count,i in enumerate(idx):
                # color_ = np.where(motion, 1-k_[count], contrast)
                x = joint_loc[i,:,0]
                y = joint_loc[i,:,1]
                plt.plot(x, y, 'o-',zorder=0.9,marker='o',color='b',lw=lw,mfc='w',
                            solid_capstyle='round', alpha= k_[count])
                plt.plot(joint_loc[i,-1,0],joint_loc[i,-1,1],'oy',markersize=3)


            
        elif task=='via2':
            ax.legend(["target-1","target-2","obstacle"])
            idx = np.arange(0,int(T/3), skip_frame)
            k_ = np.linspace(0.3,0.7,len(idx))[::-1]
            k_[-1]=1
            for count,i in enumerate(idx):
                # color_ = np.where(motion, 1-k_[count], contrast)
                x = joint_loc[i,:,0]
                y = joint_loc[i,:,1]
                plt.plot(x, y, 'o-',zorder=0.9,marker='o',color='g',lw=lw,mfc='w',
                            solid_capstyle='round', alpha= k_[count])
                plt.plot(joint_loc[i,-1,0],joint_loc[i,-1,1],'og', markersize=3)

            idx = idx = np.arange(int(T/3),2*int(T/3), skip_frame)
            k_ = np.linspace(0.1,0.2,len(idx))
            k_[-1] = 1
            for count,i in enumerate(idx):
                # color_ = np.where(motion, 1-k_[count], contrast)
                x = joint_loc[i,:,0]
                y = joint_loc[i,:,1]
                plt.plot(x, y, 'o-',zorder=0.9,marker='o',color='r',lw=lw,mfc='w',
                            solid_capstyle='round', alpha= k_[count])
                plt.plot(joint_loc[i,-1,0],joint_loc[i,-1,1],'or',markersize=3)

            idx = idx = np.arange(2*int(T/3),T, skip_frame)
            k_ = np.linspace(0.1,0.2,len(idx))
            k_[-1] = 1
            for count,i in enumerate(idx):
                # color_ = np.where(motion, 1-k_[count], contrast)
                x = joint_loc[i,:,0]
                y = joint_loc[i,:,1]
                plt.plot(x, y, 'o-',zorder=0.9,marker='o',color='k',lw=lw,mfc='w',
                            solid_capstyle='round', alpha= k_[count])
                plt.plot(joint_loc[i,-1,0],joint_loc[i,-1,1],'ok',markersize=3)

        elif task=='reaching':
            ax.legend(["target","obstacle"])
            idx = np.arange(0,int(T), skip_frame)
            k_ = np.linspace(0.3,0.7,len(idx))[::-1]
            k_[0]=1
            for count,i in enumerate(idx):
                # color_ = np.where(motion, 1-k_[count], contrast)
                x = joint_loc[i,:,0]
                y = joint_loc[i,:,1]
                plt.plot(x, y, 'o-',zorder=0.9,marker='o',color='g',lw=lw,mfc='w',
                            solid_capstyle='round', alpha= k_[count])
                plt.plot(joint_loc[i,-1,0],joint_loc[i,-1,1],'oy', markersize=3)
        elif task=='ik':
            ax.legend(["target","obstacle"])
            idx = np.arange(0,int(T), skip_frame)
            for count,i in enumerate(idx):
                # color_ = np.where(motion, 1-k_[count], contrast)
                x = joint_loc[i,:,0]
                y = joint_loc[i,:,1]
                plt.plot(x, y, 'o-',zorder=0.9,marker='o',color='g',lw=lw,mfc='w',
                            solid_capstyle='round', alpha=alpha )
                plt.plot(joint_loc[i,-1,0],joint_loc[i,-1,1],'oy', markersize=3,alpha=alpha)


        for count,i in enumerate(idx_highlight):
            color_ = [0.1]*3
            x = joint_loc[i,:,0]
            y = joint_loc[i,:,1]
            plt.plot(x, y, 'o-',zorder=0.9,marker='o',color='k',lw=lw,mfc='w',
                        solid_capstyle='round', alpha=0.5)

    plt.plot(0,0,color='y',marker='o', markersize=15)
    plt.grid(True)

    if not title is None:
        plt.title(title)
    if not save_as is None:
        fig.savefig('./images/'+save_as+".jpeg",bbox_inches='tight', pad_inches=0.01, dpi=300)

    return plt



###########################################################################
###########################################################################

def plot_point_mass(x_t, xmax=1, x_obst=[],r_obst=[],batch=False,title=None, save_as=None,figsize=3):

    fig = plt.figure(edgecolor=[0.1,0.1,0.1])
    fig.set_size_inches(figsize, figsize)

    # fig.patch.set_facecolor('white')
    # fig.patch.set_alpha(0.9)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(-xmax, xmax), ylim=(-xmax, xmax))

    if not x_obst is None:
        for i,x in enumerate(x_obst):
            circ = plt.Circle(x,r_obst[i],color='grey',alpha=0.5)
            ax.add_patch(circ)
    # if not rect_patch is None:
    #     rect = plt.Rectangle(rect_patch[0:2],rect_patch[2],rect_patch[3], color='c',alpha=0.5)
    #     ax.add_patch(rect)
    

    ax.plot(x_t[:,0,0],x_t[:,0,1],'og', markersize=10)
    ax.plot(x_t[:,-1,0],x_t[:,-1,1],'or', markersize=10)

    for i in range(x_t.shape[0]):
        plt.plot(x_t[i,:,0],x_t[i,:,1],'-b')

    # ax.legend(["target","init","obstacle"])
    plt.grid("True")

    if not title is None:
        plt.title(title)
    if not save_as is None:
        fig.savefig(save_as+".jpeg",bbox_inches='tight', pad_inches=0.01, dpi=300)

    return plt