###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import re
import numpy as np
import pandas as pd



# Create a folder for the zip file to be downloaded


'''
Define a function to track the download of the file
Report function tracks the % of download completed, 
this function is called from the download_file function
'''

# Download dictionary
def downloadFile(url):
    import zipfile,requests,os,time
    data_dir = "data" 
    start = time.clock()
    curr_dir=os.getcwd()
    #Check if the folder already exists, if the folder doesn't exist, create directory
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    #Change directory to folder where the file will be downloaded and unzipped
    os.chdir(data_dir)
    file_name = url.split('/')[-1]
    extension = url.split('.')[-1]
    #os.system('wget %s' %url)
    print "\nDownloading %s from %s" % (file_name,url)
    try:
        resp = requests.get(url)
        print "\n"
        with open(file_name, 'wb') as output:
        #    for data in tqdm(resp.iter_content(32*1024), total=file_size, unit='B', unit_scale=True):

            output.write(resp.content)
        end = time.clock()
        print('File download complete in {0:.2f}s\n'.format(end-start))
        
        if extension=="zip":
            print("Downloaded file is a zip file, unzipping...\n")
            with zipfile.ZipFile(file_name) as f:
                for files in f.infolist():
                    f.extract(files)
                    print("File %s has been unzipped\n" % (file_name))
        print("Downloaded is available in the following directory:")
        print(os.getcwd())
        
    except:
        print("\nERROR: File could not be downloaded/unzipped\n")
    os.chdir(curr_dir)
    return


from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

####################################################################################
##Function to Create Dummy Variables for Categorical Variables
####################################################################################
#def create_dummy_df(df,col,na_flag=False):
    
#    dummy_var=pd.get_dummies(df[col],prefix=[col],dummy_na=na_flag)
#    df_dummified = pd.concat([df,dummy_var],axis=1)
#    df_dummified=df_dummified.drop([col],axis=1)
#    return df_dummified

####################################################################################
##Function to Create Dummy Variables for Categorical Variables
####################################################################################


def create_dummy_df(df,colList,na_flag=False):
    df_dummified=df.copy()
    print "Following Variables will be dummified:\n",
    for col in colList:
        if df[col].dtype == np.dtype('O'):
            print col
            dummy_var=pd.get_dummies(df[col],prefix=col,dummy_na=na_flag)
            df_dummified = pd.concat([df_dummified,dummy_var],axis=1)
            df_dummified=df_dummified.drop([col],axis=1)
    return df_dummified




def create_missing_flag(df,col):
    df['Missing_Flag'+'_'+col]=df[col].isnull().astype(int)
    return df

def fix_loan_status(df):
    '''
    Fix Loan Status
    '''
    cleanLoanData = df.copy()
    cleanLoanData=cleanLoanData[(cleanLoanData["loan_status"] == "Fully Paid") |(cleanLoanData["loan_status"] == "Charged Off")]
    loan_status_dict = {"loan_status":{ "Fully Paid": 0, "Charged Off": 1}}
    cleanLoanData = cleanLoanData.replace(loan_status_dict)
    #cleanLoanData['Title'] = cleanLoanData['Name'].map(title_dict)
    return cleanLoanData


def fix_int_rate(x):
    if pd.isnull(x):
        return 0
    else:
        x = re.compile(r'[^\.0-9]*').sub("", x).strip()   
        #x=x.str.rstrip("%").astype("float")
    #x=x.replace('%','')
    return float(x)

def fix_zip_code(x):
    x = re.compile(r'[^\.0-9]*').sub("", x).rstrip('x')
    return x
 
def fix_term(x):
    x = re.compile(r'[^\.0-9]*').sub("", x).strip()
    return int(x)

#def fix_emp_length(x):
#    x = re.compile(r'[^\.0-9]*').sub("", x)
#    if x == '':
#        return np.nan
#    else:
#        return float(x)

def fix_date(df,x):
    cleanLoanData = df.copy()
    cleanLoanData[x+"_month"], cleanLoanData[x+"_year"] = zip(*cleanLoanData.x.str.split('-'))
#    df.drop(['x'], 1, inplace=True)
    return cleanLoanData


    
def fix_emp_length(df):
    cleanLoanData = df.copy()    
    cleanLoanData.replace('< 1 year', '0', inplace=True)
    cleanLoanData.replace('n/a', np.nan, inplace=True)
    cleanLoanData.emp_length.fillna(value=0, inplace=True)
    cleanLoanData['emp_length'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
    cleanLoanData['emp_length'] = cleanLoanData['emp_length'].astype(int)
    return cleanLoanData
#data['emp_length'] = data.emp_length.str.replace('+','')
#data['emp_length'] = data.emp_length_clean.str.replace('<','')
#data['emp_length'] = data.emp_length_clean.str.replace('years','')
#data['emp_length_'] = data.emp_length_clean.str.replace('year','')
#data['emp_length'] = data.emp_length_clean.str.replace('n/a','0')
#import re
#regex_pat = re.compile(r'\s+')#, flags=re.IGNORECASE)
#loan_2['emp_length_clean'] = loan_2.emp_length_clean.str.replace(regex_pat,'')


# Define Function
def data_info(df,name):
    print("Dataset **%s** has the following description: " %(name))
    #nrows, ncols = data.shape
    #print('Number of rows: {}, number of columns: {}'.format(nrows, ncols))
    print("There are {0} rows and {1} fields\n".format(*df.shape))


cat_columns = ['home_ownership', 'verification_status', 'term', 'purpose', \
                   'grade', 'sub_grade', 'issue_month', 'issue_year' \
                   , 'addr_state'] 
####################################################################################
##Function to Check Nulls
####################################################################################

def check_nulls_dups(df):
    #Check for null values in  dataframe
    print "Number of Columns with atleast one null value:"
    print pd.DataFrame(len(df.index)-df.count()[(len(df.index)-df.count())>0]).shape[0]
    
    pd.DataFrame(df.columns).to_csv("all_col.csv")

    #Export to_csv due to large size
    pd.DataFrame(len(df.index)-df.count()).to_csv("na_train_count_by_f.csv")

    print "\nNumber of Rows with atleast one column as null:"
    print df[df.isnull().any(axis=1)].shape[0]

    ####################################
    #Check for Dups
    ####################################
    #search = pd.DataFrame.duplicated(train_df)
    #print search[search == True]

    print "\nShape of Original Data: ",df.shape
    print "Shape of De-Dup Data: ",df.drop_duplicates().shape
    
    print "\nNumber of numerical columns with more than 10% nulls:\n",(df.describe().ix['count']<(df.count().max()*(1-.1))).value_counts()
    return


####################################################################################
#Data Type and Sanity Checks on Categorical Data
####################################################################################
'''
Define a function to describe the categorical attributes in the data
Output: Unique values and counts by unique values sorted in descending order
Argument: pandas dataframe
'''

def describe_cat_data(df,target):
    #Obtain Non-Numeric/Float Columns and their respective Column Counts
#print train_df[[col for col,col_type in train_df.dtypes.iteritems()if col_type<>'float64']].head()
    for col,col_type in df.dtypes.iteritems():
        if col_type=='object' and col<>target:
            print("\n**********************************")
            #print df[col].value_counts().reset_index()
            #.sort([0,'index'],ascending=[False,True]).head()
            #unique gives na but value_count by default does not
            #return NAs, hence added dropna=False
            
            a_group_desc = df[[col,target]].groupby(col).describe()
            unstacked = a_group_desc  # .unstack()
            print unstacked.loc[:,(slice(None),[
                            'count','min','mean','std','max']),]
        #print train_df.groupby(col)['target'].transform(np.sum) 
        #Doesn't work because of NAs
        
def one_hot_encode(df, cat_cols):
    # addr_state
    dummy_df = pd.get_dummies(df[cat_cols])
    df = pd.concat([df, dummy_df], axis=1)
    df = df.drop(cat_cols, axis=1)
    
    return df
    
    
def classifaction_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = re.split('[ ]+', line)
        row['class'] = row_data[1]
        row['precision'] = float(row_data[2])
        row['recall'] = float(row_data[3])
        row['f1_score'] = float(row_data[4])
        row['support'] = float(row_data[5])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    return dataframe


from sklearn.metrics import confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


from sklearn import metrics
def measure_performance(X,y,clf, show_accuracy=True, show_classification_report=True, show_confusion_matrix=True):
    y_pred=clf.predict(X)   
    if show_accuracy:
        print "Accuracy:{0:.3f}".format(metrics.accuracy_score(y,y_pred)),"\n"

    if show_classification_report:
        print "Classification report"
        print metrics.classification_report(y,y_pred),"\n"
        
    if show_confusion_matrix:
        print "Confusion matrix"
        print metrics.confusion_matrix(y,y_pred),"\n"
        



def waterfall_chart(Profit_Retained,Profit_Lost,Prevented_Loss,Additional_Loss):
#http://pbpython.com/waterfall-chart.html
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    #Use python 2.7+ syntax to format currency
    def money(x, pos):
        'The two args are the value and tick position'
        return "${:,.0f}".format(x)
    formatter = FuncFormatter(money)


    #Data to plot. Do not include a total, it will be calculated
    #index = ['sales','returns','credit fees','rebates','late charges','shipping']
    #data = {'amount': [350000,-30000,-7500,-25000,95000,-7000]}

    index = ['Profit_Retained','Profit_Lost','Prevented_Loss','Additional_Loss']
    #data = {'amount': [32199349,-24944881,77996623,-59366696]}
    data = {'amount': [Profit_Retained,Profit_Lost,Prevented_Loss,Additional_Loss]}
    trans = pd.DataFrame(data=data,index=index)

    #Store data and create a blank series to use for the waterfall
    trans = pd.DataFrame(data=data,index=index)
    blank = trans.amount.cumsum().shift(1).fillna(0)

    #Get the net total number for the final element in the waterfall
    total = trans.sum().amount
    trans.loc["net"]= total
    blank.loc["net"] = total

    #The steps graphically show the levels as well as used for label placement
    step = blank.reset_index(drop=True).repeat(3).shift(-1)
    step[1::3] = np.nan

    #When plotting the last element, we want to show the full bar,
    #Set the blank to 0
    blank.loc["net"] = 0

    #Plot and label
    my_plot = trans.plot(kind='bar', stacked=True, bottom=blank,legend=None, figsize=(10, 10), title="Profit/Loss Waterfall")
    my_plot.plot(step.index, step.values,'k')
    my_plot.set_xlabel("Profit/Loss")

    #Format the axis for dollars
    my_plot.yaxis.set_major_formatter(formatter)

    #Get the y-axis position for the labels
    y_height = trans.amount.cumsum().shift(1).fillna(0)

    #Get an offset so labels don't sit right on top of the bar
    max = trans.max()
    neg_offset = max / 25
    pos_offset = max / 50
    plot_offset = int(max / 15)

    #Start label loop
    loop = 0
    for index, row in trans.iterrows():
        # For the last item in the list, we don't want to double count
        if row['amount'] == total:
            y = y_height[loop]
        else:
            y = y_height[loop] + row['amount']
        # Determine if we want a neg or pos offset
        if row['amount'] > 0:
            y += pos_offset
        else:
            y -= neg_offset
        my_plot.annotate("{:,.0f}".format(row['amount']),(loop,y),ha="center",clip_on=True)
        loop+=1

    #Scale up the y axis so there is room for the labels
    my_plot.set_ylim(-160000000,blank.max()+int(plot_offset)*2)
    #my_plot.set_ylim(-100000000,100000000)
    #Rotate the labels
    my_plot.set_xticklabels(trans.index,rotation=0)
    #my_plot.get_figure().savefig("waterfall.png",dpi=200,bbox_inches='tight')


def model_profit(model,df,features):
    
    pred=model.predict(df[features])
    df["pred"] = pred
    pd.crosstab(df.loan_status,df.pred)
    df["net_returns"] = df["total_pymnt"] - df["loan_amnt"]
    #print(df[['net_returns',"total_pymnt","loan_amnt"]].quantile([.05,.25, .5, .75,.95,.99]))
    grouped = df.groupby(["loan_status", "pred"])
    net = pd.DataFrame(grouped.sum(col = "net_returns", na = "ignore")).reset_index()
    Profit_Retained = net[(net["loan_status"] == 0) & (net["pred"] == 0)]["net_returns"].max()
    Profit_Lost = net[(net["loan_status"] == 0) & (net["pred"] == 1)]["net_returns"].max()
    Prevented_Loss = net[(net["loan_status"] == 1) & (net["pred"] == 1)]["net_returns"].max()
    Additional_Loss = net[(net["loan_status"] == 1) & (net["pred"] == 0)]["net_returns"].max()
    waterfall_chart(Profit_Retained,-1*Profit_Lost,-1*Prevented_Loss,Additional_Loss)
    # Calculate the amount earned
    print "Profit retained by approving good lenders : %s" % '${:0,.0f}'.format(Profit_Retained)
    print "Profit lost due to declining good lenders Profit lost : %s" % '${:0,.0f}'.format(Profit_Lost)
    print "Loss  prevented by not lending to defaulters: %s" % '${:0,.0f}'.format(Prevented_Loss)
    print "Additional loss due to approving defaulters: %s" % '${:0,.0f}'.format(Additional_Loss)

    # Calculate Net
    print "Total profit by implementing model : $ %s" %'${:0,.0f}'.format((Profit_Retained - Profit_Lost + (-1*Prevented_Loss) - (-1*Additional_Loss)))
    return
