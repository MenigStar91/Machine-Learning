import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

# clusters visualization
def plot_ellipse(splot, mean, cov, color):
    v, w = linalg.eigh(cov)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1]/u[0])
    angle = 180 * angle / np.pi # convert to degrees
    ell = plt.patches.Ellipse(mean, 4 * v[0] ** 0.5, 4 * v[1] ** 0.5, 180 + angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.5)
    splot.add_artist(ell)
    splot.plot(mean[0],mean[1],"*",color="yellow",markersize=15)
    return 

# guassian distribution visualization
class M:
  def __init__ (self, data, label):

    means= data.groupby(by=label).mean()

    covar = data.groupby(by=label).cov()
    classes = label.value_counts().index
    covar.index = covar.index.get_level_values(0)

    self.means_ = pd.Series(dtype = 'float64')
    self.covariance_ = pd.Series(dtype = 'float64')

    for class_ in classes:
      self.means_.loc[class_] = np.array(means.loc[class_, :])
      self.covariance_.loc[class_] = np.array(covar.loc[class_, :])
    return

def plot_guassian(model, data, label, same_covariance=False):
    splot = plt.figure()
    splot = plt.subplot(1, 1, 1)

    cols = data.columns
    color = {}
    classes  = label.value_counts().index
    i=0
    while i<len(classes):
        c = tuple(np.random.choice(range(0, 2), size=3))
        if (c!=(0,0,0) and c!=(1,1,1) and (c not in color.values())):
            color[classes[i]]=c
            i+=1
    for class_ in classes:
        splot.scatter( data[label==class_][cols[0]] , data[label==class_][cols[1]], label = class_, c=color[class_] )
        if same_covariance:
            plot_ellipse(splot, model.means_[class_], model.covariance_, color[class_])
        else:
            plot_ellipse(splot, model.means_[class_], model.covariance_[class_], color[class_])
    
    splot.legend()
    splot.set_xlabel(cols[0])
    splot.set_ylabel(cols[1])
    return 

# decision boundry
def decision_boundry(model, data, label):
    splot = plt.figure()
    splot = plt.subplot(1, 1, 1)

    cols = data.columns
    color = {}
    classes  = label.value_counts().index
    i=0
    while i<len(classes):
        c = tuple(np.random.choice(range(0, 2), size=3))
        if (c!=(0,0,0) and c!=(1,1,1) and (c not in color.values())):
            color[classes[i]]=c
            i+=1

    for class_ in classes:
        splot.scatter( data[label==class_][cols[0]] , data[label==class_][cols[1]], label = class_, c=color[class_] )
    splot.legend()
    splot.set_xlabel(cols[0])
    splot.set_ylabel(cols[1])

    # decision boundary 
    min1, min2 = data.min() -1
    max1, max2 = data.max() +1

    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)

    xx, yy = np.meshgrid(x1grid,x2grid)
    X_grid = np.c_[xx.ravel(), yy.ravel()]

    probabilities_qda = model.predict_proba(X_grid)
    for class_ in classes:
        proba = probabilities_qda[:, class_].reshape(xx.shape)
        plt.contour(xx, yy, proba, [0.5], linewidths=2., colors='k')
    return