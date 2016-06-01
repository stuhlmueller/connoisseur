---
layout: model
title: Learning Multivariate Linear Functions
---

First, we need a loss function that compares actual and desired function output.

We use the squared-error loss:

~~~~
var squaredError = function(m1, m2) {
  var x = T.add(m1, T.mul(m2, -1));
  return T.sumreduce(T.mul(x, x));
};

var f = function(x){
  return squaredError(Vector([1, 2]), Vector([x, 2]));
};

map(f, [1, 2, 3, 4]);
~~~~

This works for transposed (column) vectors as well:

~~~~
var squaredError = function(m1, m2) {
  var x = T.add(m1, T.mul(m2, -1));
  return T.sumreduce(T.mul(x, x));
};

var f = function(x){
  return squaredError(
    T.transpose(Vector([1, 2])), 
    T.transpose(Vector([x, 2])));
};

map(f, [1, 2, 3, 4]);
~~~~

Taken together, we can try to learn multi-dimensional linear functions from data. We'll try to learn a matrix that outputs the two numbers it gets as inputs, and also their sum. That is, our desired output looks like this:

~~~~
Matrix([
    [1, 0, 1],
    [0, 1, 1]
]);
~~~~

Let's first try to do this using MCMC (with proposals from the prior):

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

var data = [
  {input: [1, 2], output: [1, 2, 3]},
  {input: [4, 5], output: [4, 5, 9]},
  {input: [0, 0], output: [0, 0, 0]},
  {input: [-2, 2], output: [-2, 2, 0]}
];

var squaredError = function(m1, m2) {
  var x = T.add(m1, T.mul(m2, -1));
  return T.sumreduce(T.mul(x, x));
};

var model = function() {
  var M = sampleGaussianMatrix([2, 3], 0, 1);
  var f = function(x) {
    return T.dot(x, M);
  }
  
  // Condition on data
  var totalError = sum(map(function(datum){
    var x = T.transpose(Vector(datum.input));
    return squaredError(f(x), Vector(datum.output));
  }, data));
  factor(-totalError);
  
  // Test
  return f(T.transpose(Vector([1, 2])));
}

var testDist = Infer(
  {method: 'MCMC', samples: 10, burn: 10000, verbose: true}, 
  model);

testDist.support();
~~~~

Unsurprisingly, the acceptance ratio is really small (0.0036).

We can approach the problem using variational inference as well. To do so, we now provide a guide distribution for the matrix prior and optimize its parameters.

~~~~
var sampleGaussianMatrix = function(dims, mean, variance, guideMean){  
  var length = dims[0] * dims[1];
  var dist = DiagCovGaussian({
    mu: Vector(repeat(length, constF(mean))),
    sigma: Vector(repeat(length, constF(variance)))
  });
  var guideMean = param([length,1], 0, 0.1);
  var guide = DiagCovGaussian({
    mu: guideMean,
    sigma: Vector(repeat(length, constF(0.001)))
  });
  var g = sample(dist, {guide: guide});
  return T.reshape(g, dims);
};

var data = [
  {input: [1, 2], output: [1, 2, 3]},
  {input: [4, 5], output: [4, 5, 9]},
  {input: [0, 0], output: [0, 0, 0]},
  {input: [-2, 2], output: [-2, 2, 0]}
];

var squaredError = function(m1, m2) {
  var x = T.add(m1, T.mul(m2, -1));
  return T.sumreduce(T.mul(x, x));
};

var model = function() {
  var M = sampleGaussianMatrix([2, 3], 0, 1, 0);
  var f = function(x) {
    return T.dot(x, M);
  }
  
  // Condition on data
  var totalError = sum(map(function(datum){
    var x = T.transpose(Vector(datum.input));
    return squaredError(f(x), Vector(datum.output));
  }, data));
  factor(-totalError);
  
  // Test
  return {
    test: f(T.transpose(Vector([1, 3]))).data,
    matrix: M.data
  };
}

var params = Optimize(model, {
  steps: 1000,
  method: {
    gd: {stepSize: 0.001}
  },
  estimator: {ELBO: {samples: 20}}});

var modelDist = SampleGuide(model, {params: params, samples: 500});

map(function(i){
  viz.auto(Enumerate(function(){return sample(modelDist).test[i];}));
}, [0, 1, 2]);

modelDist.support()[0].matrix
~~~~

The resulting matrix looks as expected.

Instead of using a factor based on the squared error, we could also factor based on a Gaussian likelihood.
