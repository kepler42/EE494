import sys
import pickle
import numpy as np
import scipy.stats as stats
from terminaltables import AsciiTable
#print(sys.stdin.read())

##################KnnC convert string to x, y coordinates##############
def str_to_xy(pred):
    pred = pred.flatten()
    return np.array([list(map(float,x_y.split(';'))) for x_y in pred])
##################KnnC convert string to x, y coordinates##############


############ Outlier Rem #################################################
def z_score(data_points):
    z_score = np.abs(stats.zscore(data_points,axis=0,nan_policy='omit'))
    print(z_score)
    outlier = (z_score < 1.6).all(axis=1) 
    print(outlier)
    return [x for x, y in zip(data_points, outlier) if y == True]

def IQR(data_points):
    Q1 = np.quantile(data_points, .25,axis=0)
    Q2 = np.quantile(data_points, .75,axis=0)

    IQR = Q2 -Q1
    b =~((data_points < (Q1 - 1.5 * IQR)) |(data_points > (Q2+ 1.5 * IQR)))
    outlier = [ True if all(l) else False for l in b  ]
    return [x for x, y in zip(data_points, outlier) if y == True]

############ Outlier Rem #################################################



def cin(load_model,load_model2,data_points,x_y):
    #function similar to cin >> data
    try:
        input_str = input()
        if input_str[0] == 'x':
            return False

        X_Y = [float(i) for i in list(input_str.split(" "))[1:3]]
        RSSI_vals = [int(i) for i in list(input_str.split(" "))[3:(NUM_CLIENTS + 3)]]
        
        data_points.append(RSSI_vals)
        x_y.append(X_Y)
        return True
    
    except:
        return False


def main():
    #C++ like main method (for illustrative/comparative purposes)
    data = [0]  
    run_var = True
    
    data_points = []
    x_y = []



    #ONLY USEFUL IF num_of_points >7
    num_of_points = 10

    filename = '/home/kepler42/EE494/EE494/Cluster/pickleRickKnn.pkl'
    filename2 = '/home/kepler42/EE494/EE494/Cluster/pickleRickKnnC.pkl'
    
    #1 - knn         2 - knnC
    load_model = pickle.load(open(filename,'rb'))
    load_model2 = pickle.load(open(filename2,'rb'))

 
    #method can be mean or centroid
    method2 = "mean"
    

    while run_var:
        run_var = cin(load_model,load_model2,data_points,x_y)


        

        if(len(data_points)%num_of_points == 0):

            knnAp = load_model.predict(data_points)
            
            knncAp = load_model2.predict(data_points)
            knncAp = str_to_xy(knncAp)

  

            distance = np.sqrt(np.sum(np.square(abs(x_y-knnAp)),1))
            distance2 = np.sqrt(np.sum(np.square(abs(x_y-knncAp)),1))

            

            data_points_corrected = IQR(data_points)
            
            #Z_score bug if all col values same causes program to crash
            #data_points_corrected2 = z_score(data_points)

            if method2 == "mean":

                #IQR
                mean_point = np.mean(data_points_corrected,axis = 0) if (len(data_points_corrected) !=0) else np.mean(data_points,axis = 0)

                result = load_model.predict([mean_point])
                result2 = load_model2.predict([mean_point])
                result2 = str_to_xy(result2)
                r1 = np.sqrt(np.sum(np.square(abs(x_y[0]-result[0]))))
                r2 = np.sqrt(np.sum(np.square(abs(x_y[0]-result2[0]))))
                
                data_points_corrected += [None]*(num_of_points-len(data_points_corrected))

                knnAp = np.around(knnAp).tolist()
                knncAp = np.around(knncAp).tolist()
                result = np.around(result).tolist()
                result2 = np.around(result2).tolist()

                table_data = [["RSSI Data","Actual Coord","knnReg Pred","Error","knnClass Pred","Error",
                "Preprocess","Pre knnReg Pred","Error","Pre knnClass Pred","Error"]]
                
                for i in range(num_of_points):
                    t = [data_points[i],x_y[i],knnAp[i],round(distance[i]),knncAp[i],
                    round(distance2[i]),data_points_corrected[i],result[0],round(r1),result2[0],round(r2)]
                    table_data.append(t)

                table = AsciiTable(table_data)
                print(table.table)

                

                # Zscore
                # print(len(data_points_corrected2))
                # if (len(data_points_corrected2) !=0):
                #     mean_point2 = np.mean(data_points_corrected2,axis = 0) 
                # else:
                #     mean_point2 =np.mean(data_points,axis = 0)

                # result3 = load_model.predict([mean_point2])
                # result4 = load_model2.predict([mean_point2])

            elif method2 == "centroid":
                predictions = load_model.predict(data_points_corrected)
                result = np.mean(predictions,axis = 0)
            else:
                raise Exception("method2 undefined")

            #print( method2 - ", method2," Acurate",result,result2)

            data_points.clear()
            x_y.clear()

NUM_CLIENTS = int(sys.argv[1])            
main()