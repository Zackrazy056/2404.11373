import numpy as np, matplotlib.pyplot as plt, warnings
from scipy.stats import gaussian_kde

def FindHeightForLevel(inArr, adLevels):

    """
    
    Function to find the height of a contour line in a 2D array.

    Parameters
    ----------

    inArr : array
        2D array of values.
    adLevels : array
        Array of levels.

    Returns
    -------
    
    adHeights : array
        Array of heights.
    
    """

    # flatten the array
    oldshape = np.shape(inArr)
    adInput= np.reshape(inArr,oldshape[0]*oldshape[1])
    # GET ARRAY SPECIFICS
    nLength = np.size(adInput)
    # CREATE REVERSED SORTED LIST
    adTemp = -1.0 * adInput
    adSorted = np.sort(adTemp)
    adSorted = -1.0 * adSorted
    # CREATE NORMALISED CUMULATIVE DISTRIBUTION
    adCum = np.zeros(nLength)
    adCum[0] = adSorted[0]
    for i in range(1,nLength):
        adCum[i] = np.logaddexp(adCum[i-1], adSorted[i])
    adCum = adCum - adCum[-1]
    # FIND VALUE CLOSEST TO LEVELS
    adHeights = []
    for item in adLevels:
        idx=(np.abs(adCum-np.log(item))).argmin()
        adHeights.append(adSorted[idx])
    adHeights = np.array(adHeights)

    return np.sort(adHeights)

def plot_contour(samples_stacked, level=[0.9], linest = 'dotted', label= None, color='k', line_w=1.2, plot_legend=1, zorder=None):

    """

    Function to plot a contour line.

    Parameters
    ----------

    samples_stacked : array
        Array of samples.
    level : array
        Array of levels.
    linest : str
        Linestyle.
    label : str
        Label.
    color : str
        Color.
    line_w : float
        Line width.
    plot_legend : int
        Plot legend.
    zorder : int
        Zorder.
    
    Returns
    -------

    Nothing, but plots a contour line.    

    """

    warnings.filterwarnings('ignore', category=RuntimeWarning)

    kde         = gaussian_kde(samples_stacked.T)
    x_flat      = np.r_[samples_stacked[:,0].min():samples_stacked[:,0].max():128j]
    y_flat      = np.r_[samples_stacked[:,1].min():samples_stacked[:,1].max():128j]
    X,Y         = np.meshgrid(x_flat,y_flat)
    grid_coords = np.append(X.reshape(-1,1),Y.reshape(-1,1),axis=1)
    pdf         = kde(grid_coords.T)
    pdf         = pdf.reshape(128,128)
    pdf[np.where(pdf==0.)] = 1.e-100

    hs  = []
    lgs = []
    for l in level:
        if zorder is not None: cntr = plt.contour(X,Y,np.log(pdf),levels = np.sort(FindHeightForLevel(np.log(pdf),[l])), colors=color, linewidths=line_w, linestyles=linest)
        else:                  cntr = plt.contour(X,Y,np.log(pdf),levels = np.sort(FindHeightForLevel(np.log(pdf),[l])), colors=color, linewidths=line_w, linestyles=linest, zorder=zorder)
        if(plot_legend):
            h,_ = cntr.legend_elements()
            hs.append(h[0])
            if not(label==None):  lgs.append(r'${0} - {1} \% \, CI$'.format(label,int(l*100.)))
            else:                 lgs.append(r'${0} \% \, CI$'.format(int(l*100.)))
    if(plot_legend): plt.legend([h_x for h_x in hs], [lg for lg in lgs], loc='upper left')

    warnings.filterwarnings('default', category=RuntimeWarning)
    
    return

Mf = [1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3]
af = [4,5,6, 1,2,3, 1,2,3, 1,2,3, 1,2,3, 1,2,3]

samples_stacked = np.column_stack((Mf, af))

plt.figure()
plot_contour(samples_stacked, [0.95, 0.5])
plt.savefig('test.png')