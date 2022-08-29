import numpy as np
def reorganize(data, win_width = 5):
    return np.hstack( [data[i:-(win_width-i)+1 or None] for i in range(win_width)] )
