---
layout: model
title: Hamiltonian Monte Carlo
---

Hamiltonian Monte Carlo is a useful method for inference in models that have many continuous random variables. An example:

~~~~
var model = function(){
  var one = sample(Gaussian({mu: 0, sigma: .5}));
  var two = sample(Gaussian({mu: one, sigma: .5}));
  var three = sample(Gaussian({mu: two, sigma: .5}));
  var four = sample(Gaussian({mu: three, sigma: .5}));
  return {
    one: one,
    four: four
  };
};

var dist = Infer({
  method:'MCMC', 
  kernel: {HMC: {steps: 20, stepSize: 0.05}}, 
  burn: 100, 
  samples: 1000}, model);

viz.auto(dist);
~~~~

However, HMC can be sensitive about the step size parameter. If we have distributions with small variance and large step size, it won't accept proposals to change the values of those distributions. For example:

~~~~
var getModel = function(sigma){
  return function(){
    var m = sample(Gaussian({mu: 0, sigma: 1}));
    var u = sample(Gaussian({mu: m, sigma: sigma}));
    return u;
  };
};

var largeSigma = getModel(1);
var smallSigma = getModel(.001);

var worksDist = Infer({
  method: 'MCMC', 
  kernel: {HMC: {steps: 20, stepSize: 0.01}}, 
  burn: 100, 
  samples: 1000}, largeSigma);
var doesntWorkDist = Infer({
  method: 'MCMC', 
  kernel: {HMC: {steps: 20, stepSize: 0.01}}, 
  burn: 100,
  samples: 1000}, smallSigma);

viz.auto(worksDist);
print(worksDist.support().length);

viz.auto(doesntWorkDist);
print(doesntWorkDist.support().length);
~~~~

This can be addressed by making the step size smaller:

~~~~
///fold:
var getModel = function(sigma){
  return function(){
    var m = sample(Gaussian({mu: 0, sigma: 1}));
    var u = sample(Gaussian({mu: m, sigma: sigma}));
    return u;
  };
};

var smallSigma = getModel(.001);
///

var worksAgainDist = Infer({
  method: 'MCMC', 
  kernel: {HMC: {steps: 20, stepSize: 0.001}}, 
  burn: 100,
  samples: 1000}, smallSigma);

viz.auto(worksAgainDist);
print(worksAgainDist.support().length);
~~~~

We might need to run the model for longer to get comparable coverage of the space.

If we have distributions that operate on different orders of magnitude and we want to use HMC, it's probably best to transform the model so that the distributions operate on similar magnitudes.