import numpy as np
class Unimodal:
    """
    symbolic dynamics of a unimodal map.
    input parameters:
         returnMap: the retur map itself
         C : critical point of this map
    """
    def __init__(self, returnMap, C):
        self.returnMap = returnMap
        self.C = C
        
    def returnMap_iter(self, x0, n):
        """
        multiple iterations of this map from point x0
        n: the iteration number.
        """
        x = x0
        future = np.array([])
        for i in range(n):
            x = self.returnMap(x)
            future = np.append(future, x)

        return future

    def future_symbol(self, x0, n):
        """
        future itinerary of point x0.
        n : the iteration number.
        """
        x = x0
        future = np.array([])
        for i in range(n):
            x = self.returnMap(x);
            if x > self.C: symbol = 1
            else: symbol = 0
            future = np.append(future, symbol)

        return future
 
