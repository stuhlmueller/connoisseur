---
layout: model
title: Learning from Weak and Strong Evidence
---

Suppose there is a latent scalar variable that we would like to learn about.

We might have an information source that provides noisy judgments (weak evidence):

~~~~
var model = function() {
  var latentQuality = uniform(0, 1);
  var noisyJudgment = beta(latentQuality*5, (1 - latentQuality)*5);
  var preciseJudgment = beta(latentQuality*50, (1 - latentQuality)*50);
  return {
    latentQuality: latentQuality,
    noisyJudgment: noisyJudgment
  };
};

var modelDist = Infer({ method: 'rejection', samples: 1000 }, model);

viz.auto(modelDist);
~~~~

Or we could have a source that provides very precise judgments (strong evidence):

~~~~
var model = function() {
  var latentQuality = uniform(0, 1);
  var noisyJudgment = beta(latentQuality*5, (1 - latentQuality)*5);
  var preciseJudgment = beta(latentQuality*50, (1 - latentQuality)*50);
  return {
    latentQuality: latentQuality,
    preciseJudgment: preciseJudgment
  };
};

var modelDist = Infer({ method: 'rejection', samples: 1000 }, model);

viz.auto(modelDist);
~~~~

If we observe a noisy judgment of .5, our posterior distribution will be fairly uncertain:

~~~~
var model = function() {
  var latentQuality = uniform(0, 1);
  var NoisyJudgment = Beta({a: latentQuality*5, b: (1 - latentQuality)*5});
  factor(NoisyJudgment.score(.5));
  return {
    latentQuality: latentQuality,
  };
};

var modelDist = Infer({ method: 'MCMC', samples: 5000, burn: 1000 }, model);

viz.auto(modelDist);
~~~~

On the other hand, if we observe a precise judgment, our posterior distribution on the latent quality will be more peaked:

~~~~
var model = function() {
  var latentQuality = uniform(0, 1);
  var PreciseJudgment = Beta({a: latentQuality*50, b: (1 - latentQuality)*50});
  factor(PreciseJudgment.score(.5));
  return {
    latentQuality: latentQuality,
  };
};

var modelDist = Infer({ method: 'MCMC', samples: 5000, burn: 1000 }, model);

viz.auto(modelDist);
~~~~
