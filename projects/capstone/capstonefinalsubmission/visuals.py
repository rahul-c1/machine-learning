###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc, roc_curve, precision_recall_fscore_support

def distribution(data, transformed = False):
    """
    Visualization code for displaying skewed distributions of features
    """
    import matplotlib.pyplot as pl
    # Create figure
    fig = pl.figure(figsize = (11,5));

    # Skewed feature plotting
    for i, feature in enumerate(['loan_amnt','annual_inc']):
        ax = fig.add_subplot(1, 2, i+1)
        ax.hist(data[feature], bins = 25, color = '#00A0A0')
        ax.set_title("'%s' Feature Distribution"%(feature), fontsize = 14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        ax.set_ylim((0, 40000))
        ax.set_yticks([0, 5000, 10000, 20000, 40000])
        ax.set_yticklabels([0, 5000, 100000, 20000, ">40K"])

    # Plot aesthetics
    if transformed:
        fig.suptitle("Log-transformed Distributions", \
            fontsize = 16, y = 1.03)
    else:
        fig.suptitle("Skewed Distributions of Continuous Data Features", \
            fontsize = 16, y = 1.03)

    fig.tight_layout()
    fig.show()

def dist_by_cat(df, column,target,**kwargs):
    '''
    Create a DataFrame of the PCA results
    Includes dimension feature weights and explained variance
    Visualizes the PCA results
    '''

    #df_for_plots_df=df.copy()
    #df_for_plots_df['Count'] =1
    #h=sns.factorplot(column,data=df,kind="count",col=target)
    #h=sns.factorplot(x=column,y='Count',hue=target,data=df_for_plots_df,palette="RdBu_r",estimator=np.sum,kind='bar',size=6,aspect=2)
    h=sns.countplot(column,hue=target,data=df,palette="RdBu_r")
    #order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    #titanic['class'].value_counts().index
    h.set_xticklabels(h.get_xticklabels(), rotation=90)


    g=sns.factorplot(x=column,y=target,data=df,size=6,kind="bar",palette="GnBu_r")
    #,order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    g.despine(left=True)
    g.set_ylabels("Event Rate")
    g.set_xticklabels(rotation=90)
    from matplotlib.ticker import FuncFormatter
    g.ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 

    return g,h


def viz_interact_factors(df,response):
    cat_col=[]
    for col in df.columns:
        if df[col].dtype == np.dtype('O'):
            cat_col.append(col)
    for pos,col in enumerate(cat_col):
        num_cat=pos
        while num_cat<len(cat_col):
            ax= sns.factorplot(x=col,y=response,hue=cat_col[num_cat],data=df,estimator=np.mean,size=6,aspect=2)
            ax.set_axis_labels(col, 'Mean(Target)')
            num_cat=num_cat+1
            ax.set_xticklabels(rotation=90)
#    from matplotlib.ticker import FuncFormatter
#    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 

    return

def viz_data(df,response):
    sns.set(style="darkgrid")   # Only need to call this once 
      # plots on same row
    cat_col=[]
    for col in df.columns:
        if df[col].dtype == np.dtype('O'):
            cat_col.append(col)
    for pos,col in enumerate(cat_col):
            f, (ax1) = plt.subplots(1,figsize=(15,6))
            #f, (ax2) = plt.subplots(1,figsize=(15,6))
            sns.set(style="darkgrid")
            #sns.boxplot(x=col, y=response, data=df,ax=ax1)
            sns.stripplot(x=col, y=response, data=df,size=4,jitter=True,edgecolor="gray",ax=ax1)

    return


def feature_plot(importances, X_train, y_train):
    import matplotlib.pyplot as pl
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Creat the plot
    fig = pl.figure(figsize = (9,5))
    pl.title("Normalized Weights for First Five Most Predictive Features", fontsize = 16)
    pl.bar(np.arange(5), values, width = 0.6, align="center", color = '#00A000', \
          label = "Feature Weight")
    pl.bar(np.arange(5) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', \
          label = "Cumulative Feature Weight")
    pl.xticks(np.arange(5), columns)
    pl.xlim((-0.5, 4.5))
    pl.ylabel("Weight", fontsize = 12)
    pl.xlabel("Feature", fontsize = 12)
    
    pl.legend(loc = 'upper center')
    pl.tight_layout()
    pl.show() 

def evaluate(results, accuracy, f1):
    """
    Visualization code to display results of various learners.
    
    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """
    import matplotlib.pyplot as pl
    import matplotlib.patches as mpatches
    import numpy as np
    import pandas as pd
    from time import time
    from sklearn.metrics import f1_score, accuracy_score

    # Create figure
    fig, ax = pl.subplots(2, 3, figsize = (11,7))

    # Constants
    bar_width = 0.3
    colors = ['#A00000','#00A0A0','#00A000']
    
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):
                
                # Creative plot code
                ax[j//3, j%3].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])
                ax[j//3, j%3].set_xticks([0.45, 1.45, 2.45])
                ax[j//3, j%3].set_xticklabels(["1%", "10%", "100%"])
                ax[j//3, j%3].set_xlabel("Training Set Size")
                ax[j//3, j%3].set_xlim((-0.1, 3.0))
    
    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")
    
    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")
    
    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[0, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    
    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color = colors[i], label = learner))
    pl.legend(handles = patches, bbox_to_anchor = (-.80, 2.53), \
               loc = 'upper center', borderaxespad = 0., ncol = 3, fontsize = 'x-large')
    
    # Aesthetics
    pl.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 16, y = 1.10)
    pl.tight_layout()
    pl.show()
    
# Whole dataset as a heatmap
def visualize_all_df(df):
    viridis = plt.cm.viridis
    viridis.set_bad((.7, 0, 0))
    fig, ax = plt.subplots(1,1)
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.name == 'category':
            df[c] = df[c].cat.codes
        if df[c].dtype.name == 'bool':
            df[c] = df[c].astype(int)
        if (df[c].max() - df[c].min()) != 0:
            df[c] = (df[c] - df[c].min())/(df[c].max() - df[c].min())

    ax.imshow(df.T, cmap=viridis)
    ax.yaxis.set_tick_params(which='minor', length=0)
    ax.set_yticks(np.array(range(len(df.columns)))+.5)
    ax.set_yticks(np.array(range(len(df.columns))), minor=True)
    ax.set_yticklabels([])
    ax.set_yticklabels(df.columns, minor=True, fontsize=9)
    ax.grid()
    fig.set_facecolor('w')
    ax.set_aspect('auto')
    fig.set_dpi(100)
    fig.set_size_inches(19.2, 10)
    fig.tight_layout()
    return fig, ax

import numpy as np
import matplotlib.pyplot as plt

def plot_validation_curve(parameter_values, train_scores, validation_scores):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)

    plt.fill_between(parameter_values, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(parameter_values, validation_scores_mean - validation_scores_std,
                     validation_scores_mean + validation_scores_std, alpha=0.1, color="g")
    plt.plot(parameter_values, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(parameter_values, validation_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.ylim(validation_scores_mean.min() - .1, train_scores_mean.max() + .1)
    plt.legend(loc="best")
 
# from figures import plot_validation_curve
#plot_validation_curve(param_range, training_scores, validation_scores)   
          
def classevaluate(results, ROCAUC, prec):
    """
    Visualization code to display results of various learners.
    
    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """
  
    # Create figure
    fig, ax = pl.subplots(2, 3, figsize = (11,7))

    # Constants
    bar_width = 0.3
    colors = ['#A00000','#00A0A0','#00A000']
    
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'ROC_AUC_train', 'precision_train', 'pred_time', 'ROC_AUC_test', 'precision_test']):
            for i in np.arange(3):
                
                # Creative plot code
                ax[j%2, j%3].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])
                ax[j%2, j%3].set_xticks([0.45, 1.45, 2.45])
                ax[j%2, j%3].set_xticklabels(["1%", "10%", "100%"])
                ax[j%2, j%3].set_xlabel("Training Set Size")
                ax[j%2, j%3].set_xlim((-0.1, 3.0))
    
    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("ROC AUC")
    ax[0, 2].set_ylabel("Precisiion")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("ROC AUC")
    ax[1, 2].set_ylabel("Precision")
    
    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("ROC AUC on Training Subset")
    ax[0, 2].set_title("Precision on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("ROC AUC on Testing Set")
    ax[1, 2].set_title("Precision on Testing Set")
    
    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y = ROCAUC, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 1].axhline(y = ROCAUC, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[0, 2].axhline(y = prec, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 2].axhline(y = prec, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    
    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color = colors[i], label = learner))
    pl.legend(handles = patches, bbox_to_anchor = (-.80, 2.53), \
               loc = 'upper center', borderaxespad = 0., ncol = 3, fontsize = 'x-large')
    
    # Aesthetics
    pl.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 16, y = 1.10)
    pl.tight_layout()
    pl.show()



def plot_roc_curve(y_test, X_test, model_dict):
    y_test_ = label_binarize(y_test, classes=[0, 1, 2])[:, :2]
    preds = {}
    fpr = {}
    tpr = {}
    roc_auc = {}
    f, ax = plt.subplots(1)
    
    #plt.figure()
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")
    
    plot_data = {}
    
    for model_key in model_dict:
        preds = model_dict[model_key].predict_proba(X_test)
        fpr = {}
        tpr = {} 
        roc_auc = {}        
        
        for i in range(2):
            fpr[i], tpr[i], _ = roc_curve(y_test_[:, i], preds[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_.ravel(), preds.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        name = "%s: (AUC = %0.2f)" % (model_key, roc_auc[1])
        plot_data = pd.DataFrame(tpr[1], index=fpr[1], columns = [name])
        plot_data.plot(ax=ax)
    plt.show()
    return ax



def linear_model_plot(model_name, train_frame):
    import re
    import numpy as np
    linear_coefs = pd.DataFrame({'value':model_name.coef_[0],
                             'param': train_frame.columns}).sort_values('value', ascending=False)
                #poly.get_feature_names(train_frame.columns)

            
    is_state = lambda x: bool(re.match('addr_state', x))
    is_grade = lambda x: bool(re.match('grade_is', x))
    is_other = lambda x: not(is_state(x) or is_grade(x))
    states = linear_coefs[linear_coefs['param'].apply(is_state)]
    grades = linear_coefs[linear_coefs['param'].apply(is_grade)]
    others = linear_coefs[linear_coefs['param'].apply(is_other)]
    f, ax = plt.subplots(1, figsize=(16, 7))
    #others.set_index('param').plot(ax=ax,kind = 'bar')
    linear_coefs.set_index('param').plot(ax=ax,kind = 'bar')
    plt.show()
    return 
    #linear_coefs
    #https://www.saltycrane.com/blog/2008/01/how-to-use-args-and-kwargs-in-python/
    #for key in kwargs:
    #    print "another keyword arg: %s: %s" % (key, kwargs[key])
    #def test_var_kwargs(farg, **kwargs):
    #print "formal arg:", farg
    #for key in kwargs:
    #   print "another keyword arg: %s: %s" % (key, kwargs[key])
    #test_var_kwargs(farg=1, myarg2="two", myarg3=3)
    #kwargs = {"arg3": 3, "arg2": "two"}
    #test_var_args_call(1, **kwargs)
    
    #def test_var_args_call(arg1, arg2, arg3):
    #print "arg1:", arg1
    #print "arg2:", arg2
    #print "arg3:", arg3
    #args = ("two", 3)    
    #test_var_args_call(1, *args)
