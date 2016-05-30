---
layout: model
title: Target and side-information but no context
author: Owain
---

### Simplest example
We try to learn a fixed constant `u` from samples of variables `zGold` and `zBias` related to `u`. The relationships are:

> zGold ~ N(u, goldSigma)

> zBias ~ N(u, biasSigma) + biasConstants

The parameters `goldSigma` and so on are all sampled independently. This means that it's possible that zBias is a less noisy function of zGold (even if it's unlikely in the prior). Alternatively we can generate `biasSigma` by adding an additional `Gamma(1,1)` to `goldSigma`, which ensure that `zBias` is more noisy.

(Note: this means that values of `zBias` provide information about `zGold` even conditioned on `u`. However, conditioned on the value of `goldSigma` and `biasSigma`, individual samples of `zBias` and `zGold` will still be independent.)

~~~~
var getMarginal = function(erp,key){
  return Enumerate(function(){
    return sample(erp)[key];
  });
};


// functional form relating u to zGold and zBias

var getZGold = function(u,params){
  return Gaussian( {mu:u, sigma:params.goldSigma} ) ;
};

var getZBias = function(u,params){
  var constant = params.biasConstant;
  var sigma = params.biasSigma;
  
  return Gaussian( {mu:u+constant, sigma:sigma} );
};


// Priors on U and params
var priorU = function(){return sample(Gaussian({mu:0, sigma:1}))};

var priorParams = function(biasDependsGold){
  if (!biasDependsGold){
    return {
      goldSigma: sample(Gamma({shape:1, scale:1})),
      biasSigma: sample(Gamma({shape:1, scale:1})),
      biasConstant: sample(Gaussian({mu:0, sigma:1}))
    };
  } else {
    var goldSigma = sample(Gamma({shape:1, scale:1}));
    var biasSigma = goldSigma + sample(Gamma({shape:1, scale:1}));

    return {
      goldSigma: goldSigma,
      biasSigma: biasSigma, 
      biasConstant: sample(Gaussian({mu:0, sigma:1}))
    };
  }
};

// Actual values U and params

var params = {
  goldSigma: .1, 
  biasConstant: .5,
  biasSigma: .2
};

var u = 1;

var getDataPoint = function(u, params){
  return {
    zGold: sample(getZGold(u,params)).toPrecision(3),
    zBias: sample(getZBias(u,params)).toPrecision(3),
  };
};

var numberDataPoints = 10;
var data = repeat( numberDataPoints, function(){getDataPoint(u,params);});

wpEditor.put('data', data);

print(JSON.stringify({actualU:u, actualParams:params}) + ' \n\n')
print('Data:  \n' + JSON.stringify({data:data}));


var conditionU = function(){
  var u = priorU();
  var params = priorParams( false );

  map(
    function(dataPoint){
      factor( getZGold(u,params).score( dataPoint.zGold) );
      factor( getZBias(u,params).score( dataPoint.zBias) );
    },
    data)

  return {u:u, 
          goldSigma: params.goldSigma,
          biasConstant: params.biasConstant,
          biasSigma: params.biasSigma
         };
};

var out = Infer({method: 'MCMC', samples: 10000}, conditionU);

// display distribution on each variable
var latents = ['u', 'goldSigma', 'biasSigma', 'biasConstant'];
map( function(latent){
  print('variable name: ' + latent);
  var erp = getMarginal(out,latent);
  viz.auto(erp);
}, latents);

~~~~

Same as above, but now `biasSigma` is at least as big as `goldSigma`. 

~~~~
var data = wpEditor.get('data');

///fold:
var getMarginal = function(erp,key){
  return Enumerate(function(){
    return sample(erp)[key];
  });
};


// functional form relating u to zGold and zBias

var getZGold = function(u,params){
  return Gaussian( {mu:u, sigma:params.goldSigma} ) ;
};

var getZBias = function(u,params){
  var constant = params.biasConstant;
  var sigma = params.biasSigma;
  
  return Gaussian( {mu:u+constant, sigma:sigma} );
};

// Priors on U and params
var priorU = function(){return sample(Gaussian({mu:0, sigma:1}))};
///


var priorParams = function(biasDependsGold){
  if (!biasDependsGold){
    return {
      goldSigma: sample(Gamma({shape:1, scale:1})),
      biasSigma: sample(Gamma({shape:1, scale:1})),
      biasConstant: sample(Gaussian({mu:0, sigma:1}))
    };
  } else {
    var goldSigma = sample(Gamma({shape:1, scale:1}));
    var biasSigma = goldSigma + sample(Gamma({shape:1, scale:1}));

    return {
      goldSigma: goldSigma,
      biasSigma: biasSigma, 
      biasConstant: sample(Gaussian({mu:0, sigma:1}))
    };
  }
};

var params = { goldSigma: .1,  biasConstant: .5, biasSigma: .2};
var u = 1;

print(JSON.stringify({actualU:u, actualParams:params}) + ' \n\n')
print('Data:  \n' + JSON.stringify({data:data}));


var biasDependsGold = true;

var conditionU = function(){
  var u = priorU();
  var params = priorParams( biasDependsGold );

  map(
    function(dataPoint){
      factor( getZGold(u,params).score( dataPoint.zGold) );
      factor( getZBias(u,params).score( dataPoint.zBias) );
    },
    data)

  return {u:u, 
          goldSigma: params.goldSigma,
          biasConstant: params.biasConstant,
          biasSigma: params.biasSigma
         };
};

var out = Infer({method: 'MCMC', samples: 10000}, conditionU);

var getMarginalPair = function(erp,key1,key2){
  return Enumerate(function(){
    var out = sample(erp);
    return {goldSigma: out[key1], 
            biasSigma: out[key2]};
  });
};

var erp2 = getMarginalPair(out,'goldSigma', 'biasSigma')
viz.auto(erp2)
~~~~

----------------

Previously u was a constant and we got a series of iid draws of `zBias` and `zGold` which depended on this fixed value. Now `u` is a random variable with a known distribution. The goal is to infer from `zBias` alone. In our training data we get to observe both `zBias` and `zGold`.

Here are samples from the generative model: 

~~~~
///fold:
var getMarginal = function(erp,key){
  return Enumerate(function(){
    return sample(erp)[key];
  });
};


// functional form relating u to zGold and zBias

var getZGold = function(u,params){
  return Gaussian( {mu:u, sigma:params.goldSigma} ) ;
};

var getZBias = function(u,params){
  var constant = params.biasConstant;
  var sigma = params.biasSigma; 
  return Gaussian( {mu:u+constant, sigma:sigma} );
};

// Priors on U and params
var priorU = function(){return sample(Gaussian({mu:0, sigma:1}))};

var priorParams = function(biasDependsGold){
  if (!biasDependsGold){
    return {
      goldSigma: sample(Gamma({shape:1, scale:1})),
      biasSigma: sample(Gamma({shape:1, scale:1})),
      biasConstant: sample(Gaussian({mu:0, sigma:1}))
    };
  } else {
    var goldSigma = sample(Gamma({shape:1, scale:1}));
    return {
      goldSigma: goldSigma,
      biasSigma: goldSigma + sample(Gamma({shape:1, scale:1})), 
      biasConstant: sample(Gaussian({mu:0, sigma:1}))
    };
  }
};
///

// Actual values U and params

var params = {
  goldSigma: .1, 
  biasConstant: .5,
  biasSigma: .2
};

var getDataPoint = function(u, params){
  return {
    u: u,
    zGold: sample(getZGold(u,params)).toPrecision(3),
    zBias: sample(getZBias(u,params)).toPrecision(3),
  };
};

var generateData = function(numberDataPoints, params){
  return repeat(numberDataPoints, 
                function(){return getDataPoint(priorU(),params);})
};                    
                   
var numberDataPoints = 500;
var data = generateData(numberDataPoints,params);

print( "u vs. zGold")
viz.scatter( _.map(data,'u'), _.map(data,'zGold') );

print( "u vs. zBias")
viz.scatter( _.map(data,'u'), _.map(data,'zBias') );

print( "zBias vs. zGold")
viz.scatter( _.map(data,'zBias'), _.map(data,'zGold') );
~~~~

We test inference. We begin with a basic 


