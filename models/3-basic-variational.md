---
layout: model
title: Basic Variational Inference
---

We'll consider the following model:

~~~~
var model = function() {
  var x = sample(Gaussian({mu: 1, sigma: 3}));
  factor(-10 * (x - 3) * (x - 3));
  return x;
};

viz.auto(SMC(model, { particles: 5000 }));
~~~~

For variational inference, we'll need to add a parameterized guide distribution:

~~~~
var model = function() {
  var mean = param(0);
  var x = sample(Gaussian({mu: 1, sigma: 3}), {
    guide: Gaussian({mu: mean, sigma: 1})
  });
  factor(-10 * (x - 3) * (x - 3));
  return x;
};

viz.auto(SMC(model, {
  particles: 5000,
  ignoreGuide: true
}));
~~~~

We can optimize the parameters for this guide distribution. In this case, we should see a guide mean close to 3:

~~~~
///fold:
var model = function() {
  var mean = param(0);
  var x = sample(Gaussian({mu: 1, sigma: 3}), {
    guide: Gaussian({mu: mean, sigma: 1})
  });
  factor(-10 * (x - 3) * (x - 3));
  return x;
};
///

var params = Optimize(model, {
  steps: 1000,
  method: {adagrad: {stepSize: 0.1}},
  estimator: 'ELBO'
});

wpEditor.put('model', model);
wpEditor.put('params', params);

params
~~~~

We can now sample from the model using a guide distribution with optimized parameters:

~~~~
var model = wpEditor.get('model');
var params = wpEditor.get('params');

viz.auto(SampleGuide(model, {params: params, samples: 500}));
~~~~