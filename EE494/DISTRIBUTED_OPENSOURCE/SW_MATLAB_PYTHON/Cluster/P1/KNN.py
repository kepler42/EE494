import numpy as np
import pandas as pd
from numpy import genfromtxt
from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import time

from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor

from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


from sklearn.metrics import explained_variance_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import max_error

from sklearn.metrics import mean_absolute_percentage_error


import pickle
# enter path to directory where data is stored
path_to_database = '/home/kepler42/EE494/DISTRIBUTED_OPENSOURCE/FINGERPRINTING_DB'

method = 'knn' # 'knn'


"""class MultiColumnLabelEncoder:

    def __init__(self, columns=None):
        self.columns = columns # array of column names to encode


    def fit(self, X, y=None):
        self.encoders = {}
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            self.encoders[col] = LabelEncoder().fit(X[col])
        return self


    def transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].transform(X[col])
        return output


    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X)


    def inverse_transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].inverse_transform(X[col])
        return output"""

def load_data(path_to_data):
    
    path_to_data = "/home/kepler42/EE494/EE494/merged.csv"
    df = pd.read_csv(path_to_data,sep=',')
    
    feature_cols = df.columns.to_list()
    rem = ['Unnamed: 0','0','1','x_y']
    keep = ['2','3','4']

    x = np.asarray(df[keep])
    print(x)

    #x_y coordinates target vars
    x_y = np.asarray(df[['0','1']] )

    if method=='knnC':
        x_y = np.tostring(np.asarray(df['0']))
        print(x_y)
        x_y = x_y.reshape(x_y.shape[0],1)


    # feature_cols = df.columns.to_list()
    # feature_cols.remove('Unnamed: 0')
    # feature_cols.remove('0')
    # feature_cols.remove('1')

    x = np.asarray(df[keep])
    # dim = np.zeros((x_y.shape[0],x_y.shape[1]+1))
    # dim[:,:-1] = x_y

    # #Converts 100 to nan
    # x[x==100] = np.nan
    print(x)

    X_train, X_test, y_train, y_test = train_test_split(x, x_y, test_size=0.20, shuffle=True)
    
    return (X_train, y_train, X_test, y_test)

#######################DISTANCE_START##############################

# Euclidean q=2
# Manhattan q=1

# Minkowski
def distance_minkowski(a, b, q):
    return np.power(np.sum(np.power(np.absolute(a-b), q)))

#######################DISTANCE_END##############################


tsum = 0
t = time.perf_counter()

# load data
X_train, y_train, X_test, y_test = load_data(path_to_database)

# y_train = y_train.ravel()
# y_test = y_test.ravel()


# prepare data for processing
ap_count = X_train.shape[1]

# #Finds floors by using Unique z vals Not used
# floors = np.unique(y_train[:,2])

#COPY OF X_train and y_train
X_ktrain = X_train.copy()
y_ktrain = y_train.copy()


if method=='knn':

    #for k in range(1,26):
        k =8
        knn = KNeighborsRegressor(n_neighbors = k, weights='distance', algorithm='kd_tree',p=3,metric='euclidean')
        regr = MultiOutputRegressor(knn)
        # regr is the model 
        regr.fit(X_ktrain, y_ktrain) 
        y_pred = regr.predict(X_test)
        print("R2 score is ", r2_score(y_test,y_pred, multioutput = 'variance_weighted')," for K-Value:",k)
    
        distance = np.sqrt(np.sum(np.square(abs(y_test-y_pred)),1))
        max_error = np.amax(distance)
        print("max_error score is ", max_error," for K-Value:",k)

        mean_error = np.sum(distance)/y_test.shape[0]
        print("mean_error score is ", mean_error," for K-Value:",k)




elif method=='knnC':
    print(y_ktrain)

#     for K in range(25):
#         K_value = K+1
        
#         knn = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto')
#         regr = MultiOutputClassifier(knn)
#         regr.fit(X_ktrain, y_ktrain)

#         # regr is the model  
#         y_pred = regr.predict(X_test)
#         print("R2 score is ", r2_score(y_test,y_pred)," for K-Value:",K_value)


else:
    print('Unknown method. Please choose either "km" or "ap".')
    quit()



#if cused.size > 0:
#    print('cused %.2lf' % np.mean(cused))
filename = '/home/kepler42/EE494/EE494/DISTRIBUTED_OPENSOURCE/SW_MATLAB_PYTHON/Cluster/P1/pickleRick.pkl'
pickle.dump(regr, open(filename,'wb'))

tsum += time.perf_counter() - t
print('\n time  %.2lf s' % tsum)

#knn = KNeighborsRegressor(n_neighbors = K_value, weights='distance', algorithm='kd_tree',p=2,metric='chebyshev') -> R2 0.67 k=8