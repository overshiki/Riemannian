## Exprimental Implementation of Riemannian manifold calculation 
This would be useful in natural gradient and other model comparision technique

### current status:
Implemented a small toy method for calculation fisher information matrix. It is slow and, only support small scale neural network since for large scales, the resulting matrix would be too large for either gpu memory or host memory 

A toy example is provided for small NeuralNet model with only 11560 parameters.

### future development:
Robust method for fisher information matrix combined with natural gradient calculation, hopefully we could use some reduce technique to reduce the memory consumption