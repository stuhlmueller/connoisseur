---
layout: model
title: Multivariate Gaussian Distributions
---

We first build a function that helps us visualize two-dimensional distributions:

~~~~
var showMultivariateDist = function(dist) {
  var f = function() {
    var s = sample(dist);
    return {
      x: s.data[0],
      y: s.data[1]
    }
  };
  var out = Infer({method: 'rejection', samples: 5000}, f);
  viz.auto(out);
};

wpEditor.put('showMultivariateDist', showMultivariateDist);
~~~~

A Gaussian with a diagonal covariance matrix looks like this:

~~~~
var showMultivariateDist = wpEditor.get('showMultivariateDist');

var dist = DiagCovGaussian({
  mu: Vector([0,0 ]), 
  sigma: Vector([5, 5])
});

showMultivariateDist(dist);
~~~~

And a general multivariate Gaussian:

~~~~
var showMultivariateDist = wpEditor.get('showMultivariateDist');

var dist = MultivariateGaussian({
  mu: Vector([0, 0]),
  cov: Matrix([[10, 5], [5, 5]])
});

showMultivariateDist(dist);
~~~~
