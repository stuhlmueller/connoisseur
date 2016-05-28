---
layout: model
title: Bayesian Neural Net
hidden: true
---

~~~~
var inputSize = 1;
var hiddenSize = 5;  // number of hidden units
var outputSize = 1;

var nnparam = function(a, b) {
  return param([a, b], 0, 0.1);
};

var network = function(x, W, b) {
  var input = T.reshape(x, [x.length, 1]);
  var h = T.tanh(T.add(T.dot(W[0], input), b[0]));
  var output = T.add(T.dot(W[1], h), b[1]);
  return T.sumreduce(output);
};

var sampleGaussianMatrix = function(dims, mean, variance, guideMean){
  var length = dims[0] * dims[1];
  var dist = DiagCovGaussian({
    mu: Vector(repeat(length, constF(mean))),
    sigma: Vector(repeat(length, constF(variance)))
  });
  var guide = DiagCovGaussian({
    mu: T.reshape(guideMean,[length,1]),
    sigma: Vector(repeat(length, constF(0.001)))
  });
  var g = sample(dist, {guide: guide});
  return T.reshape(g, dims);
};


var model = function() {

  var W0guide = nnparam(hiddenSize, inputSize);
  var W1guide = nnparam(1, hiddenSize);
  var b0guide = nnparam(hiddenSize, 1);
  var b1guide = nnparam(1, 1);

  var W0 = sampleGaussianMatrix([hiddenSize, inputSize], 0, 10, W0guide);
  var W1 = sampleGaussianMatrix([1, hiddenSize], 0, 10, W1guide);
  var b0 = sampleGaussianMatrix([hiddenSize, 1], 0, 10, b0guide);
  var b1 = sampleGaussianMatrix([1, 1], 0, 10, b1guide);

  var f = function(x){
    return network(Vector([x]), [W0, W1], [b0, b1]);
  };
  
  var inputs = [1, 2, 3, 4, 5];
  var outputs = [3, 5, 7, 9, 11];
  var score = sum(map2(function(input, output) {
    console.log(input, f(input).x);
    return Gaussian({mu: f(input), sigma: 1}).score(output);
  }, inputs, outputs));
  factor(score);
};

var params = Optimize(model, {
  steps: 2000,
  method: {
    gd: {stepSize: 0.001}
  },
  estimator: {ELBO: {samples: 1}}});

params
~~~~