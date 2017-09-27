import csv
import optunity
import sklearn.metrics
from sklearn.svm import SVC
import numpy as np
from sklearn.decomposition import PCA 
#import matplotlib.pyplot as plt

def load_csv(filename):
    data = []

    # open file
    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        # load as a list of instances
        for row in reader:
            data.append(row)
    # done!
    return data

def load_labels(filename):
    data = []

    # open file
    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        # load as a list of instances
        for row in reader:
            data.append(str(row[0]))
    # done!
    return data


class scoreModel:
    def __init__(self, model, scoring_method):
        self.model = model
        self.scoring_method = scoring_method    #str of name of scoring method to use
        
    def score_model(self, x_train, y_train, x_test, y_test, **params):
        model_inst = self.model(**params)
        model_inst.fit(x_train, y_train)
        decision_vals = model_inst.predict(x_test)
        scoring = getattr(sklearn.metrics, self.scoring_method)
        score = scoring(y_test, decision_vals)
        return score

    def hyp_opt_optunity(self, data, labels, p, num_folds, n_evals, **kwargs):
        if p:   #parallelisation option
            return 0
        else:
            return self.optimize_optunity(data, labels, n_evals, num_folds, **kwargs)
    
    def optimize_optunity(self, data, labels, n_evals, num_folds, **kwargs):
        cv_decorator = optunity.cross_validated(x=data, y=labels, num_folds=num_folds)
        scoreModel = cv_decorator(self.score_model)
        optimisedHyp = optunity.maximize(scoreModel, num_evals=n_evals, **kwargs)
        return optimisedHyp



def compute_pca(hyperparam_scores):    
    #hyperparam_scores: dict of args and score values of each arg combo
        
    n_hyp = len(hyperparam_scores['values'])    
    X = [[] for i in range(n_hyp)]
    
    n=0
    for i in range(len(hyperparam_scores['values'])):
        for k,v in hyperparam_scores['args'].items(): 
            X[n].append(hyperparam_scores['args'][k][i])
        n+=1    

    pca = PCA(n_components = 2)
    pca.fit(X)
    X_transform = pca.transform(X)
    x_coord_pca = X_transform[:, 0]
    y_coord_pca = X_transform[:, 1]
    eigenvals = pca.components_     
    
    #plt.scatter(x_coord_pca, y_coord_pca)    
    return {'x_coord_pca': x_coord_pca, 'y_coord_pca': y_coord_pca, 'corrMatrix': eigenvals}
    


data = load_csv("abalone.data")
labels = load_labels("abalone.labels")

model= scoreModel(SVC, 'accuracy_score')
results = model.hyp_opt_optunity(data, labels, False, 5, 5, C=[5, 7], gamma=[0, 1])
opt_hyperparams = [results[0], results[1][0] ]
hyperparam_scores = results[1][2]
pca_report = compute_pca(hyperparam_scores)



'''    
#For Principal Component = 3

from mpl_toolkits.mplot3d import Axes3D

def plot_figs(fig_num, elev, azim):
    fig = plt.figure(fig_num, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=elev, azim=azim)

    ax.scatter(a[::10], b[::10], c[::10], c=density[::10], marker='+', alpha=.4)
    Y = np.c_[a, b, c]

    # Using SciPy's SVD, this would be:
    # _, pca_score, V = scipy.linalg.svd(Y, full_matrices=False)

    pca = PCA(n_components=3)
    pca.fit(Y)
    pca_score = pca.explained_variance_ratio_
    V = pca.components_

    x_pca_axis, y_pca_axis, z_pca_axis = V.T * pca_score / pca_score.min()

    x_pca_axis, y_pca_axis, z_pca_axis = 3 * V.T
    x_pca_plane = np.r_[x_pca_axis[:2], - x_pca_axis[1::-1]]
    y_pca_plane = np.r_[y_pca_axis[:2], - y_pca_axis[1::-1]]
    z_pca_plane = np.r_[z_pca_axis[:2], - z_pca_axis[1::-1]]
    x_pca_plane.shape = (2, 2)
    y_pca_plane.shape = (2, 2)
    z_pca_plane.shape = (2, 2)
    ax.plot_surface(x_pca_plane, y_pca_plane, z_pca_plane)
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
'''


