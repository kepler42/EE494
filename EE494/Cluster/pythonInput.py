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
    threshold =1.7

    filename = '/home/kepler42/EE494/EE494/Cluster/pickleRick.pkl'
    load_model = pickle.load(open(filename,'rb'))

    #mehtod can be mean or centroid
    method = "mean"
    
    while run_var:
        run_var = cin(load_model,data_points)
        if(len(data_points)%num_of_points == 0):
            z_score = np.abs(stats.zscore(data_points,axis=0))
            outlier = (z_score < threshold).all(axis=1) 
            data_points_corrected = [x for x, y in zip(data_points, outlier) if y == True]
            print(data_points_corrected)

            if method == "mean":
                mean_point = np.mean(data_points_corrected,axis = 0)
                result = load_model.predict([mean_point])
                print("Acurate  ",result)
            # elif method == "centroid":


            
main()