---
layout: model
title: Priors on Multivariate Linear Functions
---

We can interpret matrices as linear functions:

~~~~
var M = Matrix([
  [1,3,1],
  [2,1,4]
]);

var input = T.transpose(Vector([1, 5]));

var f = function(x) {
  return T.dot(x, M);
}

f(input);
~~~~

We can sample Gaussian matrices:

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

wpEditor.put('sampleGaussianMatrix', sampleGaussianMatrix);

sampleGaussianMatrix([2, 3], 0, 1);
~~~~

Thus, we have a prior on multivariate linear functions:

~~~~
var sampleGaussianMatrix = wpEditor.get('sampleGaussianMatrix');

var functionPrior = function() {
  var matrix = sampleGaussianMatrix([2, 3], 0, 1);
  var f = function(x) {
    return T.dot(x, matrix);
  }
  return f;
}

var f = functionPrior();
var input = T.transpose(Vector([1, 5]));
f(input);
~~~~