import numpy as np
from numpy.linalg import inv
import sys

if __name__ == "__main__":
    mat = [ [-3     ,1.25   ,0.5],
            [1.25   ,-1     ,-0.5],
            [0.5    ,-0.5   ,-4]]
    mat = np.matrix([   [-3     ,1.25   ,0.5],
                        [1.25   ,-1     ,-0.5],
                        [0.5    ,-0.5   ,-4]],dtype=np.float64)
    print((np.round(mat)).astype(dtype=np.int32))

    print(np.iinfo(np.int32).max)
    print(np.iinfo(np.int32).min)
    
    pass