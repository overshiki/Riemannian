## Exprimental Implementation of Riemannian manifold calculation 
This would be useful in natural gradient and other model comparision technique

### current status:
Implemented a small toy method for calculation fisher information matrix. It is slow and, only support small scale neural network since for large scales, the resulting matrix would be too large for either gpu memory or host memory 

### usage:
see example.py file