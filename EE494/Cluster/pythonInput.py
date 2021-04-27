#import sys
#print(sys.stdin.read())
import pickle
import numpy as np
import scipy.stats as stats



def cin(load_model,data_points):
    #function similar to cin >> data

    try:
        input_str = input()
        if input_str[0] == 'x':
            return False
        RSSI_vals = [int(i) for i in list(input_str.split(" "))[1:4]]
    
        data_points.append(RSSI_vals)
        print(data_points)
        result = load_model.predict([RSSI_vals])
       
        print("Approx  ",result)
        
        return True
    
    except:
        return False

def main():
    #C++ like main method (for illustrative/comparative purposes)
    data = [0]  
    run_var = True
    
    data_points = []
    #Vars

    #ONLY USEFUL IF num_of_points >7
    num_of_points = 10
    threshold =1.6

    filename = '/home/kepler42/EE494/EE494/Cluster/pickleRick.pkl'
    load_model = pickle.load(open(filename,'rb'))

    #method can be std or IQR
    method1 = "std"
    #method can be mean or centroid
    method2 = "mean"
    
    while run_var:
        run_var = cin(load_model,data_points)
        if(len(data_points)%num_of_points == 0):

            if method1 == "std":
                z_score = np.abs(stats.zscore(data_points,axis=0))
                outlier = (z_score < threshold).all(axis=1) 
                data_points_corrected = [x for x, y in zip(data_points, outlier) if y == True]
                print(data_points_corrected)

            elif method1 == "IQR":
                Q1 = np.quantile(data_points, .25,axis=0)
                Q2 = np.quantile(data_points, .75,axis=0)

                IQR = Q2 -Q1
                outlier1 =~((data_points < (Q1 - 1.5 * IQR)) |(data_points > (Q2+ 1.5 * IQR)))
                outlier2 = [ True if all(l) else False for l in outlier1 ]
                data_points_corrected = [x for x, y in zip(data_points, outlier2) if y == True]
            else:
                raise Exception("method1 undefined")


            if method2 == "mean":
                mean_point = np.mean(data_points_corrected,axis = 0)
                result = load_model.predict([mean_point])
            elif method2 == "centroid":
                predictions = load_model.predict(data_points_corrected)
                result = np.mean(predictions,axis = 0)
            else:
                raise Exception("method2 undefined")
            print("method1 - ",method1, "method2 - ", method2," Acurate",result)

            data_points.clear()
            
            
main()