---
layout: model
title: Learning about the Reliability of Information Sources
---

We can learn about the reliability (evidence strength) of information sources without ever observing ground truth judgments (given strong assumptions about possible kinds of noise):

~~~~
var data = [
  { alice: [.5, .55, .5], bob: [.3, .4, .5] },
  { alice: [.2, .21, .19], bob: [.1, .2, .01] },
  { alice: [.7, .71, .67], bob: [.7, .5, .9] }
];

var getDatum = function(bobReliability, aliceReliability) {
  var latentQuality = uniform(0, 1);
  var bobJudgmentDist = Beta({
    a: latentQuality*bobReliability,
    b: (1 - latentQuality)*bobReliability});
  var aliceJudgmentDist = Beta({
    a: latentQuality*aliceReliability,
    b: (1 - latentQuality)*aliceReliability});
  return {
    latentQuality: latentQuality,
    bobJudgmentDist: bobJudgmentDist,
    aliceJudgmentDist: aliceJudgmentDist
  };
};

var model = function() {
  var bobReliability = uniform(0, 100);
  var aliceReliability = uniform(0, 100);
  map(function(datum){
    var hypDatum = getDatum(bobReliability, aliceReliability);
    map(function(x){factor(hypDatum.aliceJudgmentDist.score(x));}, datum.alice);
    map(function(x){factor(hypDatum.bobJudgmentDist.score(x));}, datum.bob);    
  }, data);
  return {
    aliceReliability: aliceReliability,
    bobReliability: bobReliability
  }
};

var modelDist = Infer({ method: 'MCMC', samples: 5000, burn: 1000 }, model);

var vizDim = function(dist, dim) {
  viz.auto(Enumerate(function(){return sample(modelDist)[dim];}))
}

vizDim(modelDist, 'aliceReliability');
vizDim(modelDist, 'bobReliability');
~~~~

Alice's inferred reliability is much higher than Bob's.