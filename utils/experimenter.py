import logging
import torch
import os
import os.path as osp
import sys
from torch import optim

from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn import svm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import LinearSVC
import sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from scipy.stats import randint, uniform
#import hdbscan
from sklearn.decomposition import PCA
import numpy as np
from scipy.optimize import linear_sum_assignment


def find_corner_simplex(simplex_size, data_points, labels):
    categorical_map = np.eye(simplex_size) #.to(args.device)
    pred_label_array = []
    for i in range(len(data_points)):
        distance_array = []
        for j in range(len(categorical_map)):
            #distance = (categorical_map[j] - data_points[i]) * (categorical_map[j] - data_points[i])
            distance = np.linalg.norm(categorical_map[j]-data_points[i])
            distance_array.append(distance)
        pred_label = np.argmin(distance_array)
        pred_label_array.append(pred_label)
    score = adjusted_mutual_info_score(labels, pred_label_array)
    print("NMI score on training data", str(score))
    return score



def hung_acc(y_true, y_predicted, cluster_number=None):
    """
    Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to
    determine reassignments.
    :param y_true: list of true cluster numbers, an integer array 0-indexed
    :param y_predicted: list  of predicted cluster numbers, an integer array 0-indexed
    :param cluster_number: number of clusters, if None then calculated from input
    :return: reassignment dictionary, clustering accuracy
    """
    if cluster_number is None:
        cluster_number = (
            max(y_predicted.max(), y_true.max()) + 1
        )  # assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size
    return  accuracy

def max_operator(simplex_size, data_points, labels, hung=None):
   
    pred_label_array = []
    for i in range(len(data_points)):
        pred_label = np.argmax(data_points[i])
        pred_label_array.append(pred_label)
    score = adjusted_mutual_info_score(labels, pred_label_array)
    
    hung_score = hung_acc(np.asarray(labels), np.asarray(pred_label_array), simplex_size)
    print("NMI score on training data {}, hung acc {} ".format(str(score), hung_score))
    if hung is not None:
        return score, hung_score
    return score

def max_operator_all(simplex_size, data_points, labels, hung=None):
   
    pred_label_array = []
    for i in range(len(data_points)):
        pred_label = np.argmax(data_points[i])
        pred_label_array.append(pred_label)
    score = adjusted_mutual_info_score(labels, pred_label_array)
    nmi = normalized_mutual_info_score(labels, pred_label_array)
    hung_score = hung_acc(np.asarray(labels), np.asarray(pred_label_array), simplex_size)
    print("AMI score on training data {}, hung acc {} NMI {}".format(str(score), hung_score, nmi))
   
    return score , hung_score, nmi


def get_clusters(n_clusters, data_points, inits=100):
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=inits)
    kmeans.fit(data_points)
    pred_labels = kmeans.labels_
    return pred_labels, kmeans

def clustering(n_clusters, data_points, labels, inits=100, hung=None):
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=inits)
    kmeans.fit(data_points)
    pred_labels = kmeans.labels_
    score = adjusted_mutual_info_score(labels, pred_labels)
    hung_score = hung_acc(labels, pred_labels, cluster_number=n_clusters)
    print("NMI score {} and hung acc {} on training data".format(score, hung_score) )
    if hung is not None:
        return score, hung_score 
    return score

def two_clustering(n_clusters, data_points, labels, inits=100):
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=inits)
    kmeans.fit(data_points)
    pred_labels = kmeans.labels_
    score = adjusted_mutual_info_score(labels, pred_labels)
    hung_score = hung_acc(labels, pred_labels, cluster_number=n_clusters)
    print("NMI score {} and hung acc {} on training data".format(score, hung_score) )
    return score, hung_score

def plot_tsne(data_points, labels=None, save_loc=None):

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data_points)
    my_colors = {0:'orange',1:'red',2:'green',3:'blue',4:'grey',5:'gold',6:'violet',7:'pink',8:'navy',9:'black'}

    if labels == None:
        plt.plot(tsne_results[:,0], tsne_results[:,1],'ro' )
    else:
        for i, data_point in enumerate(tsne_results):
            plt.scatter(data_point[0] , data_point[1], color = my_colors.get(labels[i], 'black'))
    
    #plt.axis('off')
    #plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    if save_loc != None:
        plt.savefig(save_loc)
    plt.show()