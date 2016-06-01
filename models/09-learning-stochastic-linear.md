---
layout: model
title: Learning Stochastic Multivariate Linear Functions
---

This is essentially the same as what we have seen previously when we learned multivariate linear functions, except now we are using the function prior that supplies its function with an additional noise input, and thus we learn conditional distributions instead of deterministic functions.

We'll also supply our functions with an additional unit input, as is usual in Bayesian regression.

As before, we will use variational inference.

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

var functionPrior = function(dims) {
  var matrixDims = [dims[0] + 2, dims[1]];
  var matrix = sampleGaussianMatrix(matrixDims, 0, 1, 0);
  var f = function(x) {
    var u = sample(Uniform({a: 0, b: 1}), { guide: Uniform({a: 0, b: 1}) });
    var input = T.transpose(Vector(x.concat([u, 1])));
    return T.dot(input, matrix);
  }
  _.assign(f, {'matrix': matrix});
  return f;
};

var squaredError = function(m1, m2) {
  var x = T.add(m1, T.mul(m2, -1));
  return T.sumreduce(T.mul(x, x));
};

var data = [
  {input: [1, 2], output: [1, 2.2, 3]},
  {input: [4, 5], output: [4, 4.6, 9]},
  {input: [0, 0], output: [0, .3, 0]},
  {input: [-2, 2], output: [-2, 1.8, 0]}
];

var model = function() {
  
  var f = functionPrior([2, 3]);
  
  // Condition on data
  var totalError = sum(map(function(datum){
    return squaredError(f(datum.input), Vector(datum.output));
  }, data));
  factor(-totalError);
  
  // Test
  return {
    test: f([1, 3]).data,
    matrix: f.matrix
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

modelDist.support()[0].matrix;
~~~~

Both the test case predictions and introspection on the learned matrix (last row) show that we correctly learned to noisify only the second output element.