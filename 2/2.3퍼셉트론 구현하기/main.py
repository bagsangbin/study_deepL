def And(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp<theta:
        return 0
    elif tmp >= theta:
        return 1
    
def __init__():
    print(And(0,0))
    print(And(0,1))
    print(And(1,0))
    print(And(1,1))

__init__()