---
layout: model
title: Priors on Functions
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

To compare actual and desired function output, we use a squared-error loss function:

~~~~
var squaredError = function(a, b) {
  var err = T.sumreduce(T.add(a, T.mul(b, -1)));
  return err * err;
}

var f = function(x){
  return squaredError(Vector([1, 2]), Vector([x, 2]));
};

map(f, [1, 2, 3, 4]);
~~~~

This works for transposed (column) vectors as well:

~~~~
var squaredError = function(a, b) {
  var err = T.sumreduce(T.add(a, T.mul(b, -1)));
  return err * err;
}

var f = function(x){
  return squaredError(
    T.transpose(Vector([1, 2])), 
    T.transpose(Vector([x, 2])));
};

map(f, [1, 2, 3, 4]);
~~~~

Taken together, we can try to learn multi-dimensional linear functions from data. We'll try to learn a matrix that outputs the two numbers it gets as inputs, and also their sum. That is, our desired output looks like this:

~~~~
var M = Matrix([
    [1, 0, 1],
    [0, 1, 1]
]);
~~~~

Let's first try to do this using MCMC (with proposals from the prior):

~~~~
var sampleGaussianMatrix = wpEditor.get('sampleGaussianMatrix');

var data = [
  {input: [1, 2], output: [1, 2, 3]},
  {input: [4, 5], output: [4, 5, 9]},
  {input: [0, 0], output: [0, 0, 0]},
  {input: [-2, 2], output: [-2, 2, 0]}
];

var squaredError = function(a, b) {
  var err = T.sumreduce(T.add(a, T.mul(b, -1)));
  return err * err;
}

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