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
path_to_database = "/home/kepler42/EE494/EE494/Cluster/merged2.csv"

method = 'knn' # 'knn'


def load_data(path_to_data):
    
    df = pd.read_csv(path_to_data,sep=',')
    
    feature_cols = df.columns.to_list()
    rem = ['Unnamed: 0','0','1','x_y']
    keep = ['2','3','4']
    # df[keep] = df[keep].apply(lambda x: np.power(10,-x/10))
    

    x = np.asarray(df[keep])
    

    #x_y coordinates target vars
    x_y = np.asarray(df[['0','1']] )

    if method=='knnC':
        df["x_y"] = (df["0"].apply(str) +";"+ df["1"].apply(str)).astype("string")
        x_y = np.asarray(df[['x_y']] )


    x = np.asarray(df[keep])

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


##################KnnC convert string to x, y coordinates##############
def str_to_xy(pred):
    pred = pred.flatten()
    return np.array([list(map(float,x_y.split(';'))) for x_y in pred])
##################KnnC convert string to x, y coordinates##############





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

    #for k in range(1,26):
        k = 8
        knn = KNeighborsClassifier(n_neighbors = k, weights='distance', algorithm='kd_tree',p=3,metric='euclidean')
        regr = MultiOutputClassifier(knn)
        regr.fit(X_ktrain, y_ktrain)
       

       # regr is the model  
        y_pred = regr.predict(X_test)
        convert_y_pred = str_to_xy(y_pred)
        convert_y_test = str_to_xy(y_test)
  

        print("R2 score is ", r2_score(convert_y_test,convert_y_pred)," for K-Value:",k)
        distance = np.sqrt(np.sum(np.square(abs(convert_y_test-convert_y_pred)),1))
        max_error = np.amax(distance)
        print("max_error score is ", max_error," for K-Value:",k)
        mean_error = np.sum(distance)/y_test.shape[0]
        print("mean_error score is ", mean_error," for K-Value:",k)


else:
    print('Unknown method. Please choose either "kcc" or "ap".')
    quit()



#if cused.size > 0:
#    print('cused %.2lf' % np.mean(cused))



filename = '/home/kepler42/EE494/EE494/Cluster/pickleRick2.pkl'
pickle.dump(regr, open(filename,'wb'))

tsum += time.perf_counter() - t
print('\n time  %.2lf s' % tsum)

#knn = KNeighborsRegressor(n_neighbors = K_value, weights='distance', algorithm='kd_tree',p=2,metric='chebyshev') -> R2 0.67 k=8