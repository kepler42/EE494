#import sys
#print(sys.stdin.read())
import pickle



def cin():
    #function similar to cin >> data

    try:
        input_str = input()
        if input_str[0] == 'x':
            return False
        RSSI_vals = [int(i) for i in list(input_str.split(" "))[1:4]]

        # result = load_model.predict([RSSI_vals])
        # print(result)
        return True
    
    except:
        return False

def main():
    #C++ like main method (for illustrative/comparative purposes)
    data = [0]  
    run_var = True
    
    c = []
    filename = '/home/kepler42/EE494/EE494/DISTRIBUTED_OPENSOURCE/SW_MATLAB_PYTHON/Cluster/P1/pickleRick.pkl'
    load_model = pickle.load(open(filename,'rb'))
    
    while run_var:
        input_str = input()
        c.append([int(i) for i in list(input_str.split(" "))[1:4]])
        print(c)
        run_var = cin()
main()