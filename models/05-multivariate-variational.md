---
layout: model
title: Variational Inference for Multivariate Distributions
---

We can add matrices like this:

~~~~
T.add(
  Matrix([[1,2,3], [4,5,6]]), 
  Matrix([[-1,-2,-3], [-4,-5,-6]]));
~~~~

We can multiply matrices by scalars like this:

~~~~
T.mul(Matrix([[1,2,3], [4,5,6]]), -1)
~~~~

We can compute the element-wise sum of a matrix like this:

~~~~
T.sumreduce(Matrix([[1,2,3], [4,5,6]]));
~~~~

We use these operations to compute the element-wise squared distance between two matrices:

~~~~
var squaredDistance = function(m1, m2) {
  var x = T.add(m1, T.mul(m2, -1));
  return T.sumreduce(T.mul(x, x));
};

wpEditor.put('squaredDistance', squaredDistance);

squaredDistance(Matrix([[1,2,3],[4,5,6]]), Matrix([[-1,2,3],[4,5,6]]));
~~~~

Now, given a prior on Gaussian matrices, we should be able to use inference to find a matrix that minimizes the distance to the given one.

Let's start by trying this using MCMC (with matrices re-sampled from the prior), and using only a 1x1-dimensional matrix (i.e. a scalar):

~~~~
var sampleGaussianMatrix = function(dims, mean, variance, guideMean){  
  var length = dims[0] * dims[1];
  var dist = DiagCovGaussian({
    mu: Vector(repeat(length, constF(mean))),
    sigma: Vector(repeat(length, constF(variance)))
  });
  var guideMean = param([length, 1], 0, 0.1);
  var guide = DiagCovGaussian({
    mu: guideMean,
    sigma: Vector(repeat(length, constF(0.001)))
  });
  var g = sample(dist, {guide: guide});
  return T.reshape(g, dims);
};

wpEditor.put('sampleGaussianMatrix', sampleGaussianMatrix);

var squaredDistance = wpEditor.get('squaredDistance');

var target = Matrix([[4]]);

var model = function() {
  var matrix = sampleGaussianMatrix([1, 1], 2, 2, 2);  
  var score = -10 * squaredDistance(matrix, target);
  factor(score);
  return matrix.data;
};

viz.auto(Infer({method: 'MCMC', samples: 5000, burn: 10000}, model));
~~~~

This works. We can apply the same approach to a 2x1-dimensional matrix:

~~~~
var sampleGaussianMatrix = wpEditor.get('sampleGaussianMatrix');
var squaredDistance = wpEditor.get('squaredDistance');

var target = Matrix([[0, 4]]);

var model = function() {
  var matrix = sampleGaussianMatrix([2, 1], 2, 2, 2);  
  var score = -squaredDistance(matrix, target);
  factor(score);
  return matrix.data;
};

var modelDist = Infer({method: 'MCMC', samples: 10000, burn: 20000}, model);

map(function(i){
  viz.auto(Enumerate(function(){return sample(modelDist)[i];}));
}, [0, 1]);
~~~~

For higher-dimensional distributions, this inference method will stop working well:

~~~~
var sampleGaussianMatrix = wpEditor.get('sampleGaussianMatrix');
var squaredDistance = wpEditor.get('squaredDistance');

var target = Matrix([[0, 1, 2], [3, 4, 5]]);

var model = function() {
  var matrix = sampleGaussianMatrix([3, 2], 2, 2, 2);  
  var score = -squaredDistance(matrix, target);
  factor(score);
  return matrix.data;
};

var modelDist = Infer({method: 'MCMC', samples: 10000, burn: 20000}, model);

map(function(i){
  viz.auto(Enumerate(function(){return sample(modelDist)[i];}));
}, [0, 1, 2, 3, 4, 5]);
~~~~

We can then apply variational inference. Using the very peaked guide distribution defined above (in `sampleGaussianMatrix`), this will result in point estimates:

~~~~
var sampleGaussianMatrix = wpEditor.get('sampleGaussianMatrix');
var squaredDistance = wpEditor.get('squaredDistance');

var target = Matrix([[0, 1, 2], [3, 4, 5]]);

var model = function() {
  var matrix = sampleGaussianMatrix([3, 2], 2, 2, 2);  
  var score = -squaredDistance(matrix, target);
  factor(score);
  return matrix.data;
};

var params = Optimize(model, {
  steps: 10000,
  method: {adagrad: {stepSize: 0.1}},
  estimator: 'ELBO'
});

var modelDist = SampleGuide(model, {params: params, samples: 5000});

map(function(i){
  viz.auto(Enumerate(function(){return sample(modelDist)[i];}));
}, [0, 1, 2, 3, 4, 5]);
~~~~