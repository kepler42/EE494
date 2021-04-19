#import sys
#print(sys.stdin.read())

def cin():
    #function similar to cin >> data
    try:
        input_str = input()
        if input_str[0] == 'x':
            return False
        RSSI_vals = [int(i) for i in list(input_str.split(" "))[1:4]]
        print(RSSI_vals)
        return True
    
    except:
        return False

def main():
    #C++ like main method (for illustrative/comparative purposes)
    data = [0]
    run_var = True

    while run_var:
        run_var = cin()
main()