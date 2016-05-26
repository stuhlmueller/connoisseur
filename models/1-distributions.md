---
layout: model
title: Sampling from Distributions
---

Bernoulli:

~~~~
var coin = Bernoulli({p: .7});

viz.auto(coin)
~~~~

Beta:

~~~~
var showDist = function(dist) {
  var hist = Infer({method: 'Rejection', samples: 1000}, function(){
    return sample(dist);
  });
  return viz.auto(hist);
};

showDist(Beta({a: 5, b: 1}));
showDist(Beta({a: 5, b: 5}));
showDist(Beta({a: 1, b: 5}));
~~~~

Beta-binomial:

~~~~
var betaBinomial = function(numHeads, numTails) {
  var p = sample(Beta({a: 1, b: 1}));
  var coin = Bernoulli({p: p});
  factor(coin.score(true) * numHeads);
  factor(coin.score(false) * numTails);
  return p;
};

var showBetaBinomial = function(numHeads, numTails) {
  viz.auto(Infer({method: 'Rejection', samples: 10000}, function(){
    return betaBinomial(numHeads, numTails);
  }));
}

showBetaBinomial(4, 0);
showBetaBinomial(4, 4);
showBetaBinomial(0, 4);
~~~~