---
layout: model
title: Inference Algorithms
---

~~~~
var result = Infer({method: 'MCMC', samples: 100}, function(){
  return uniform(0, 1);
});

viz.auto(result);
~~~~