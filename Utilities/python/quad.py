import math
def quad(*xs):
   '''
   Function to sum in quadrature, avoid replication everywhere
   '''
   return math.sqrt(sum(x*x for x in xs))
