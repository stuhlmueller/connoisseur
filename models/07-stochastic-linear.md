---
layout: model
title: Priors on Stochastic Multivariate Linear Functions
---

We can concatenate vectors as follows:

~~~~
var v = Vector([1,2]);
var x = Vector([3]);

T.concat(v, x)
~~~~

We can generate a stochastic multivariate linear function simply by concatenating a uniformly random input to our input vector, and by extending the matrix dimension appropriately:

~~~~
var sampleGaussianMatrix = function(dims, mean, variance){
  var length = dims[0] * dims[1];
  var dist = DiagCovGaussian({
    mu: Vector(repeat(length, constF(mean))),
    sigma: Vector(repeat(length, constF(variance)))
  });
  var g = sample(dist);
  return T.reshape(g, dims);
};

var functionPrior = function(dims) {
  var matrixDims = [dims[0] + 1, dims[1]];
  var matrix = sampleGaussianMatrix(matrixDims, 0, 1);
  var f = function(x) {
    var u = uniform(0, 1);
    var input = T.transpose(Vector(x.concat(u)));
    return T.dot(input, matrix);
  }
  return f;
};

var f = functionPrior([2, 3]);
repeat(10, function(){
  print(f([1, 5]).data);
});
~~~~