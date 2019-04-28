# Enter your code here. Read input from STDIN. Print output to STDOUT

# import numpy as np 
# no numpy

import random

def play(roll, current, la_st, la_end, sn_st, sn_end):
    if current + roll <= 100:
        current += roll 
        if current in la_st:
            current = la_end[la_st.index(current)]
        elif current in sn_end:
            # print(current)
            current = sn_st[sn_end.index(current)]
            # print(current)
    return current


# def rolls(cumpr):
#     ru = random.uniform(0,1)
#     if ru < cumpr[0]:
#         roll = 1
#     elif ru < cumpr[1]:
#         roll = 2
#     elif ru < cumpr[2]:
#         roll = 3
#     elif ru < cumpr[3]:
#         roll = 4
#     elif ru < cumpr[4]:
#         roll = 5
#     else:
#         roll = 6    
#     return roll
        
def simulations(p, la, sn, la_st_end, sn_st_end, nit):
    la_st = [el[0] for el in la_st_end]
    la_end = [el[1] for el in la_st_end]
    # print(la_end)
    sn_st = [el[0] for el in sn_st_end]
    sn_end = [el[1] for el in sn_st_end]
    # print(sn_end)

    for i in range(1, 6):
        p[i] += p[i-1]
    # print(p)
    totalrols = 0
    totalrolnum = 0
    for i in range(nit):
        # print(i)
        # generate rand unif in 1-6 with the prob
        current = 1
        Nrolls = 0
        # roll
        while Nrolls < 1000:
            ru = random.uniform(0,1)
            roll = 0
            if ru < p[0]:
                roll = 1
            elif ru < p[1]:
                roll = 2
            elif ru < p[2]:
                roll = 3
            elif ru < p[3]:
                roll = 4
            elif ru < p[4]:
                roll = 5
            else:
                roll = 6    
            # print('roll', roll)
            # current = play(roll, current, la_st, la_end, sn_st, sn_end)
            if current + roll <= 100:
                current += roll 
                if current in la_st:
                    current = la_end[la_st.index(current)]
                elif current in sn_st:
                # print(current)
                    current = sn_end[sn_st.index(current)]
            # print(current)
            # print('current', current)
            Nrolls += 1
            # print('Nrolls', Nrolls)
            if current == 100:
                totalrols += Nrolls
                totalrolnum += 1
                break
                    
    return totalrols / totalrolnum


def main():
    # read in data
    n = int(input())
    for i in range(n):
        p = list(map(float, input().split(',')))
        #print(p)
        la, sn = list(map(int, input().split(',')))
        #print(la, sn)
        la_st_end = [tuple(map(int,el.split(','))) for el in input().split(' ')]
        sn_st_end = [tuple(map(int, el.split(','))) for el in input().split(' ')]
        #print(sn_st_end)
        print(round(simulations(p, la, sn, la_st_end, sn_st_end, nit=5000)))

if __name__ == '__main__':
    main()

