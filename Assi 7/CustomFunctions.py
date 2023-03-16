'''Assi 5 Functions'''
import numpy as np
import numpy.matlib as nm
# import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import seaborn as sb

def parity_plot(y_test, y_hat, xlabel = 'actual value', ylabel = 'predicted value', title = None, subplot=False):
    '''
    input:
    y_test: the actual y values
    y_hat: the predicted y values
    xlabel
    ylabel

    output:
    parity plots

    Improvements: 
    1. make R= value appear as a label, check how PRA did it;
    2. enable subplot, still looks like a hassel'''

    if not subplot: # if subplot is False, 
        plt.figure(figsize=(10, 10)) #create a stand-alone figure

    plt.scatter(y_test, y_hat, c='b', s=4, alpha = 0.4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    r1 = np.arange(0,1,0.1)[:,None]
    r2 = r1
    r2value = round(r2_score(y_test, y_hat),3)
    plt.plot(r1, r2, c='r', label='$R^{2}$='+ str(r2value))
    # make the legend in the proper place
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.title(title)



def poly_predict(X, y, X_test, n_degree = 3, return_y_train_hat = False, return_model = False):
    '''this function takes the raw training set of X and y, and produce a prediction
    steps involved:
    1. process X and X_test following the same procedure (polynomialize, standardize).
    2. fit the processed X set and y, and use that fit to predict y from X_test
    
    it can be used by itself or inside the loop in the CV functions below'''

    poly = PolynomialFeatures(degree=n_degree, include_bias=False)
    std = StandardScaler()
    RG = LinearRegression(fit_intercept=True)
    
    X_poly, X_test_poly = poly.fit_transform(X), poly.fit_transform(X_test)

    ### Set the std to fit the training set
    std.fit(X_poly)

    ### and apply the std to fit to training AND test X. 
    ### Finalized train and test sets are called X_train and X_test
    X_train, X_test = std.transform(X_poly), std.transform(X_test_poly)

    ### fit linear regression to training set
    RG.fit(X_train, y) 
    
    ### predict the test set
    y_hat = RG.predict(X_test)
    
    ### we can also return the prediction on the training set, if requested
    if return_y_train_hat:
        y_train_hat = RG.predict(X_train)
        return y_hat, y_train_hat
    if return_model:
        return y_hat, RG
    else:
        return y_hat
    
def random_cv(df, n_degree = 3, n_split = 5):
    '''plot parity plots for random train-test split'''
    # first, grab the X, y from df
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    ### Define a condition which loops through N_split times
    for i in range(1, n_split+1):

        # then using train_test_split, split into test and val set
        X_tr, X_te, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

        '''The code below is identical to the LOCO_cv function'''
        y_hat, y_train_hat = poly_predict(X_tr, y_train, X_te, n_degree)

        ### make 10 subplots, first row training error, second row validation error
        plt.figure(2*n_split, figsize=(20,9))

        plt.subplot(2,n_split,i)
        parity_plot(y_train, y_train_hat, title = 'Train Err, Split #'+ str(i), subplot = True)

        ### make validation R2 plots
        plt.subplot(2,n_split, i+5)
        parity_plot(y_test, y_hat, title = 'Val Err, Split #'+ str(i), subplot = True)

def LOCO_cv(df, n_degree = 3, element_order = ['Co', 'Ti', 'Cr']):

    '''this function does mainly two things: 
    Task 1. slice the df into train-test sets by ternary systems, therefore establish the folds for LOCO-CV
    Task 2. borrows the poly_predict function to report training and validation performance needed for parity plot
    '''
    # Perform LOCO-cv on each ternary system
    for e in element_order:
        i = element_order.index(e)+1
        ### use i to index the position of the subplot
        
        df_tr = df[df[e] != 0]   # set those with a certain element as train set,
        df_te = df[df[e] == 0]    # and those without as test
        # Note: can do the other way, but this ensures size(train)>size(test)

        ### define which columns are X and which are y, for both train and test set
        y_train = df_tr.iloc[:, -1].values
        y_test = df_te.iloc[:, -1].values

        X_tr = df_tr.iloc[:, :-1].values
        X_te = df_te.iloc[:, :-1].values

        '''end of Task 1'''

        ### use the handy function to get the y_hat and y_train_hat (for this split)
        y_hat, y_train_hat = poly_predict(X_tr, y_train, X_te, n_degree)
        
        ### make six subplots, first row training error, second row validation error
        plt.figure(6, figsize=(17,11))

        ### make training R2 plots
        plt.subplot(2,3,i)
        parity_plot(y_train, y_train_hat, title = 'train err, split by '+e, subplot = True)

        ### make validation R2 plots
        plt.subplot(2,3,i+3)
        parity_plot(y_test, y_hat, title = 'val err, split by '+e, subplot = True)

def supervised_drop(df, corr_threshold=0.85, must = [], exception = [], color_scheme = 'YlOrBr', heatmap_annote=True, update_df = False):
    '''the function takes a dataframe that is solely numerical. 
    It plots the heatmap of the correlation matrix before and after dropping highly correlated features.'''

    # do a correlation analysis on df, drop the most correlated features
    corr = df.corr().abs()

    # drop the diagonal and upper triangle
    corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # find the columns with correlation > threshold, return a list of columns
    to_drop = [column for column in corr.columns if any(corr[column] > corr_threshold)]

    # if must is specified, we must drop them
    for m in must:
        if m not in to_drop and m in df.columns:
            to_drop.append(m)

    # if exceptions are specified, we do not drop them
    for e in exception:
        if e in to_drop:
            to_drop.remove(e)

    # create a new df with correlated features dropped
    df_dropped = df.drop(columns=to_drop)
    
    # same as above, but for the new df
    corr_new = df_dropped.corr().abs()
    # drop the diagonal and upper triangle
    corr_new = corr_new.where(np.triu(np.ones(corr_new.shape), k=1).astype(bool))

    # plot the heatmaps
    # set a fig with two subplots, arranged side by side
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    fig.subplots_adjust(wspace=0.01)

    # plot the heatmaps. 
    ### .T is just to put the remaining matrix on the bottom, for better readability
    sb.heatmap(corr.T, cmap=color_scheme, ax = ax1, annot=heatmap_annote)
    sb.heatmap(corr_new.T, cmap=color_scheme, ax=ax2, annot=heatmap_annote)
    sb.set(rc={'figure.figsize':(20,16)})
    plt.tight_layout()
    plt.show()

    ### if choose to update, drop correlated features directly in the original df 
    if update_df == True:
        df = df_dropped 
    
    return df, to_drop

def plot_yang(df, plot_title, color_label = None, colorbar = True, cmap = 'viridis'):
    '''This function takes a dataframe with at least the following columns:
    Yang omega, Yang delta, and MaxFWHM. 
    
    It plots the Yang delta vs Yang omega, and color the scatter by MaxFWHM
    '''
    
    # We can choose to color the scatter by either:
    # if color_label is anything other than None, color by color_label
    # else, color by MaxFWHM

    if color_label is not None:
        color = color_label # OR by color label determined from clustering
    else:
        color = df['MaxFWHM'] # by default (MaxFWHM) 
    
    element = plt.scatter(df['Yang delta'], df['Yang omega'], 
                          s = 5, c=color, alpha = 1, cmap=cmap)
    # set the y axis to log scale
    plt.yscale('log')
    # set the title, x and y labels
    plt.title(plot_title)
    plt.xlabel('Yang delta')
    plt.ylabel('Yang omega')
    # ax.colorbar(mappable = None, label='Predicted MaxFWHM', ax=ax)

    if colorbar:
        plt.colorbar(element)

def pca_plot(X, title, n_comp, y_log = False, color_label = None, return_scree = False):
    '''This function takes and array X, plots the first two principal components, 
    and plots the explained variance ratio as a scree plot.

    It returns the dataset that is cut down to 'n_comp' principal components.'''

    # standardize the data
    X = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=n_comp)
    pca.fit(X)
    pca_dots = pca.transform(X)

    # this part of PCA is only for plotting Scree Plot
    pca_scree = PCA(n_components=X.shape[1])
    pca_scree.fit(X)
    pca_dots = pca.transform(X)

    # make two subplots, one for pca_dots and one for the explained variance ratio
    if return_scree:
        plt.figure(2, figsize=(10,4))
        plt.subplot(1,2,2) 
        ### now let's see what its explained variance looks like
        plt.plot(pca_scree.explained_variance_ratio_, 'o-')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Scree Plot')
        plt.subplot(1,2,1)

    if color_label is not None:
        color = color_label # OR by color label determined from clustering
    else:
        color = 'k'

    # map the datapoint in PCA space
    plt.scatter(pca_dots[:,0], pca_dots[:,1], s = 6, c=color, marker='o', alpha=0.3)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    
    if y_log:
        plt.yscale('log')
    
    return pca_dots

import ternary
import matplotlib

def plot_ternary(data, components, z, label, title='', cmap=plt.cm.nipy_spectral):
    
    fig, ax = plt.subplots()
    
    scale = 100

    grid = plt.GridSpec(10, 10, wspace=2, hspace=1)
    ax = plt.subplot(grid[:,:9])
    
    figure, tax = ternary.figure(scale=scale, ax=ax)
    # Draw Boundary and Gridlines
    tax.boundary(linewidth=2.0)
    tax.gridlines(color="blue", multiple=10)

    # Set Axis labels and Title
    fontsize = 12
    offset = 0.14
    tax.right_corner_label(components[0], fontsize=fontsize, offset=0.2, fontweight='bold')
    tax.top_corner_label(components[2], fontsize=fontsize, offset=0.23, fontweight='bold')
    tax.left_corner_label(components[1], fontsize=fontsize, offset=0.2, fontweight='bold')
    tax.left_axis_label("at.%", fontsize=fontsize, offset=offset)
    tax.right_axis_label("at.%", fontsize=fontsize, offset=offset)
    tax.bottom_axis_label("at.%", fontsize=fontsize, offset=offset)
    tax.ticks(axis='lbr', multiple=10, linewidth=1, offset=0.025, clockwise= True)
    tax.get_axes().axis('off')
    tax.clear_matplotlib_ticks()
    
    
    # Create color map and plot color bar
    cmap = cmap
    
    norm = plt.Normalize(0, z.max())
    
    tax.scatter(data, marker='o', c=cmap(norm(z)), edgecolors='k', alpha=1, s=20, vmin=z.min(), vmax=z.max())
    
    ax = plt.subplot(grid[1:-1,9:])
    cb1 = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical', label=label)
    cb1.set_label(label=label, size=18)


def df_plot_ternary(df, z, label, ax, title='', cmap=plt.cm.nipy_spectral):
    '''take a three-column df (3 element compositions),
    the color label (z) and label for the colorbar'''
    # make a variable called component which is a list of the first three columns' names
    components = list(df.columns[:3])

    cmp = df.loc[:,components].to_numpy()
    # multiply by 100 to get at.% composition
    cmp = cmp*100
    points=nm.vstack((cmp[:,0].T,cmp[:,2].T)).T

    # fig, ax = plt.subplots()
    
    scale = 100

    grid = plt.GridSpec(10, 10, wspace=2, hspace=1)
    # ax = plt.subplot(grid[:,:9])
    
    figure, tax = ternary.figure(scale=scale, ax=ax)
    # Draw Boundary and Gridlines
    tax.boundary(linewidth=2.0)
    tax.gridlines(color="blue", multiple=10)

    # Set Axis labels and Title
    fontsize = 12
    offset = 0.14
    tax.right_corner_label(components[0], fontsize=fontsize, offset=0.2, fontweight='bold')
    tax.top_corner_label(components[2], fontsize=fontsize, offset=0.23, fontweight='bold')
    tax.left_corner_label(components[1], fontsize=fontsize, offset=0.2, fontweight='bold')
    tax.left_axis_label("at.%", fontsize=fontsize, offset=offset)
    tax.right_axis_label("at.%", fontsize=fontsize, offset=offset)
    tax.bottom_axis_label("at.%", fontsize=fontsize, offset=offset)
    tax.ticks(axis='lbr', multiple=10, linewidth=1, offset=0.025, clockwise= True)
    tax.get_axes().axis('off')
    tax.clear_matplotlib_ticks()
    
    
    # Create color map and plot color bar
    cmap = cmap
    
    norm = plt.Normalize(0, z.max())
    
    tax.scatter(points, marker='o', c=cmap(norm(z)), 
                # edgecolors='k', 
                alpha=1, s=20, vmin=z.min(), vmax=z.max())
    
    ax2 = plt.subplot(grid[1:-1,9:])
    cb1 = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, orientation='vertical', label=label)
    cb1.set_label(label=label, size=9)






    '''ASSIGNMENT 7 FUNCTIONS'''
from sklearn.model_selection import StratifiedShuffleSplit, LeaveOneGroupOut
def parity_plotCV(estimator, X, y, split_by, cv_label = None, title = ''):
    '''plot a series of parity plot for cross validation
    estimator: the model
    X: the features
    y: the target
    split_by: initialized split method, such as ShuffleSplit
    cv_label: data label, in case of stratified split'''
    if isinstance(split_by, LeaveOneGroupOut):
        n = split_by.get_n_splits(X, y, groups = cv_label)
    else:
        n = split_by.get_n_splits(X, y) 

    for i, (train_index, test_index) in enumerate(split_by.split(X, y, cv_label)): 
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # fit the model
        estimator.fit(X_train, y_train)

        # predict on test and train set
        y_hat = estimator.predict(X_test)
        y_train_hat = estimator.predict(X_train)

        i+=1
        # plot the parity plot
        plt.figure(2*n, figsize=(20, 40/n))

        plt.subplot(2,n,i)
        parity_plot(y_train, y_train_hat, title = 'Train Err, Split #'+ str(i), subplot = True)

        ### make validation R2 plots
        plt.subplot(2,n, i+n)
        parity_plot(y_test, y_hat, title = 'Val Err, Split #'+ str(i), subplot = True)

    plt.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.show()