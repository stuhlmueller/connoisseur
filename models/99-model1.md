---
layout: model
title: Model 1---Utility directly observed, Bias side-info depends on utility
author: Owain
---

### Introduction
Model 1 has the following form:

> X is a vector random variable (as in supervised learning), the "context". X is always fully observed and we don't model its prior distribution.

> u:X -> R, where u ~ prior(u)

> b: R->R, where b ~ prior(b) and b is stochastic

> Training data has form: (x_i, u(x_i), b(u(x_i))), where i is the index.

> Test data has form: (x_i, ___, b(u(x_i))), where the goal is to infer u(x_i).


We are interested in cases with the following additional properties:

- The training data is insufficient for us to learn `u`. That is, on some regions of X, we will be quite uncertain about u(x).

- Our test data will include x-values from some such regions. If the training data shows `b` to be generally informative about `u`, then conditioning on `b` will improve our posteriors over `u` in those regions. (If `b` is not informative, then conditioning should not help us).

So we will compare algorithms that condition on the `b`-values at test time with algorithms that do not (and simply do supervised learning using the training data). We use examples where `u` is piecewise linear with different parameters in different regions. We begin with examples where `b` is not a function of X (i.e. the relation between the output of `u` and `b` does not depend directly on the context X).

### Example 1

Here is the generative model. We make both `u` and `b` stochastic functions. 

~~~~
var getDatumGivenContext = function(x, params){
  var u = params.u;
  var b = params.b;
  var ux = sample(u(x));
  var bx = sample(b(ux));
  return {x:x, u:ux, b: bx};
};
~~~~

We pick some parameters for `u` and `b` and then generate some data. Note that `u` is a piecewise linear with boundaries at -1 and 1. The function `b` is adds a constant and some Gaussian noise to `u(x)`. 

~~~~

var u = function(x){
  
  var getCoefficient = function(x){
    if (x<-1){return -1.5}
    if (x<1){return 3}
    if (x>=1){return -.3}
  }
    
  return Gaussian({mu:getCoefficient(x)*x, sigma:.1})
}

var b = function(ux){
    var constant = .5
    var sigma = .1
    return Gaussian({mu:ux + constant, sigma:sigma});
}


var displayFunctions = function(u,b){
    print('Display true functions u and b on range [-3,3]');

    var xs = repeat(100,function(){return sample(Uniform({a:-3,b:3}))});
    var uValues =  map(function(x){return sample(u(x))}, xs);
    viz.scatter(xs, uValues);
    var bValues =  map(function(ux){return sample(b(ux))}, uValues);
    viz.scatter(xs, bValues);
};

displayFunctions(u,b)

var trueParams = {u:u, b:b};
wpEditor.put('trueParams', trueParams);
~~~~

We generate some data. The training and test data have different distributions. This means that `u` is not learnable from the training data but its values can be inferred fairly accurately given that `b` is informative about `u`.


~~~~

///fold:
var getDatumGivenContext = function(x, params){
  var u = params.u;
  var b = params.b;
  var ux = sample(u(x));
  var bx = sample(b(ux));
  return {x:x, u:ux, b: bx};
  };
///


var getAllData = function(numberTrainingData, numberTestData, params){

  var generateData = function(numberDataPoints, priorX){
    var xs = repeat(numberDataPoints, priorX);
    
    return map(function(x){
      return getDatumGivenContext(x, params);
    }, xs); 
  };
  
  var trainingPrior = function(){return sample(Uniform({a:-.5, b:.5}));}
  var trainingData = generateData(numberTrainingData, trainingPrior)
  
  var testPrior = function(){return sample(Uniform({a:-1.5, b:1.5}));}
  var testData = generateData(numberTestData, testPrior) 
  
  var displayData = function(trainingData, testData){
    print('Display Training and then Test Data')
     
    map(function(data){
      var xs = _.map(data,'x');
      var bs = _.map(data,'b');
      viz.scatter(xs,bs)
    }, [trainingData, testData]);
  };
  
  displayData(trainingData, testData)
  
  return {params: params,
          trainingData: trainingData,
          testData: testData};
};

var trueParams = wpEditor.get('trueParams');
var allData = getAllData(10, 20, trueParams);
wpEditor.put('allData', allData);
~~~~

To perform inference, we need a prior on the functions `u` and `b`. We simply abstract out the parameters that we used to define the functions `u` and `b` above:

~~~~
var getU = function(uParams){
  return function(x){
  
  var getCoefficient = function(x){
    if (x<-1){return uParams.lessMinus1}
    if (x<1){return uParams.less1}
    if (x>=1){return uParams.greater1}
  }
    
  return Gaussian({mu:getCoefficient(x)*x, sigma:.1})
  }
}

var getB = function(bParams){
  return function(ux){
    return Gaussian({mu:ux + bParams.constant, sigma:bParams.sigma});
  }
};

var priorParams = function(){
  var sampleG = function(){return sample(Gaussian({mu:0, sigma:2}));}
  var uParams = {lessMinus1: sampleG(),
  less1: sampleG(),
  greater1: sampleG()}

  var bParams = {constant: sampleG(), sigma: Math.abs(sampleG())}

  return {
    u: getU(uParams),
    uParams: uParams,
    b: getB(bParams),
    bParams: bParams
    }
}

var displayFunctions = function(u,b){
    print('Display functions u and b on range [-3,3]');

    var xs = repeat(100,function(){return sample(Uniform({a:-3,b:3}))});
    var uValues =  map(function(x){return sample(u(x))}, xs);
    viz.scatter(xs, uValues);
    var bValues =  map(function(ux){return sample(b(ux))}, uValues);
    viz.scatter(xs, bValues);
};
    
var paramsExample = priorParams()
displayFunctions(paramsExample.u, paramsExample.b)
~~~~

Putting everything together, we do inference on the training set to learn the parameters of u and b. We first do this using likelihoods (via `score`) and then we do this using an error function. 

~~~~
///fold:

var u = function(x){
  
  var getCoefficient = function(x){
    if (x<-1){return -1.5}
    if (x<1){return 3}
    if (x>=1){return -.3}
  }
    
  return Gaussian({mu:getCoefficient(x)*x, sigma:.1})
}

var b = function(ux){
    var constant = .5
    var sigma = .1
    return Gaussian({mu:ux + constant, sigma:sigma});
}


var displayFunctions = function(u,b){
    print('Display true functions u and b on range [-3,3]');

    var xs = repeat(100,function(){return sample(Uniform({a:-3,b:3}))});
    var uValues =  map(function(x){return sample(u(x))}, xs);
    viz.scatter(xs, uValues);
    var bValues =  map(function(ux){return sample(b(ux))}, uValues);
    viz.scatter(xs, bValues);
};

displayFunctions(u,b)
var trueParams = {u:u, b:b};


var getDatumGivenContext = function(x, params){
  var u = params.u;
  var b = params.b;
  var ux = sample(u(x));
  var bx = sample(b(ux));
  return {x:x, u:ux, b: bx};
  };


var getAllData = function(numberTrainingData, numberTestData, params){

  var generateData = function(numberDataPoints, priorX){
    var xs = repeat(numberDataPoints, priorX);
    
    return map(function(x){
      return getDatumGivenContext(x, params);
    }, xs); 
  };
  
  var trainingPrior = function(){return sample(Uniform({a:-.5, b:.5}));}
  var trainingData = generateData(numberTrainingData, trainingPrior)
  
  var testPrior = function(){return sample(Uniform({a:-1.5, b:1.5}));}
  var testData = generateData(numberTestData, testPrior) 
  
  var displayData = function(trainingData, testData){
    print('Display Training and then Test Data')
     
    map(function(data){
      var xs = _.map(data,'x');
      var bs = _.map(data,'b');
      viz.scatter(xs,bs)
    }, [trainingData, testData]);
  };
  
  displayData(trainingData, testData)
  
  return {params: params,
          trainingData: trainingData,
          testData: testData};
};

var allData = getAllData(10, 20, trueParams);



var getU = function(uParams){
  return function(x){
  
  var getCoefficient = function(x){
    if (x<-1){return uParams.lessMinus1}
    if (x<1){return uParams.less1}
    if (x>=1){return uParams.greater1}
  }
    
  return Gaussian({mu:getCoefficient(x)*x, sigma:.1})
  }
}

var getB = function(bParams){
  return function(ux){
    return Gaussian({mu:ux + bParams.constant, sigma:bParams.sigma});
  }
};

var priorParams = function(){
  var sampleG = function(){return sample(Gaussian({mu:0, sigma:2}));}
  var uParams = {lessMinus1: sampleG(),
  less1: sampleG(),
  greater1: sampleG()}

  var bParams = {constant: sampleG(), sigma: Math.abs(sampleG())}

  return {
    u: getU(uParams),
    uParams: uParams,
    b: getB(bParams),
    bParams: bParams
    }
}
///


var trainingModel = function(){
  var params = priorParams();
  var u = params.u
  var b = params.b
  
  map( function(datum){
    factor( u(datum.x).score(datum.u) );
    factor( b(datum.u).score(datum.b) );
  }, allData.trainingData);
  
  return {
    uParams: params.uParams,
    bParams: params.bParams,
  };
};


var distance = function(x,y){return Math.abs(x-y)};

var trainingModelError = function(){
  var params = priorParams();
    
  map( function(datum){
    var prDatum = getDatumGivenContext(datum.x,params)
    var error = distance(prDatum.u, datum.u) + distance(prDatum.b, datum.b)
    factor(-error)
  }, allData.trainingData);
  
  return {
    uParams: params.uParams,
    bParams: params.bParams,
  };
};

var getPosterior = function(model){

    var posterior = Infer(
        {method:'MCMC', 
   kernel:{HMC: {steps:10, stepSize:.1}},
   burn:1, 
   samples:1000,
  }, 
  model);

var MAP = posterior.MAP().val
print( '\nTrue vs. MAP uParams:' +
JSON.stringify({lessMinus1: -1.5, less1:3, greater1:-.3}) +
JSON.stringify(MAP.uParams) +
'\n\nTrue vs. MAP bParams:' +
JSON.stringify({constant: .5, sigma:.1}) +
JSON.stringify(MAP.bParams));

return posterior;
}

print('Model with likelihoods')
getPosterior(trainingModel)

print('Model with absolute error')
var posteriorTraining = getPosterior(trainingModelError)


var predictModelError = function(posteriorTraining){
  return function(){
    var trainingParams = sample(posteriorTraining)
    var u = getU(trainingParams.uParams)
    var b = getB(trainingParams.bParams)
    var params = {u:u, b:b}
    
    var datumToError = map( function(datum){
      var prDatum = getDatumGivenContext(datum.x,params)
      factor(-distance(prDatum.b, datum.b))
    
    var error = distance(prDatum.u, datum.u)
    return error;
      
    }, allData.testData);
    
    var totalError = sum( datumToError)
  
    return {
      uParams: trainingParams.uParams,
      bParams: trainingParams.bParams,
      totalError: totalError
    };
  };
}

var predictModel = predictModelError(posteriorTraining);

var posteriorPredict = Infer(
  {method:'MCMC', 
   //kernel:{HMC: {steps:10, stepSize:.1}},
   burn:10, 
   samples:1000,
  }, 
  predictModel);

print( JSON.stringify(posteriorPredict.MAP().val ))
var marg = Infer({method:'rejection',
    samples:200}, function(){
  return sample(posteriorPredict).totalError
}) 
viz.auto(marg)

~~~~


The results show that the model does not infer the correct parameters for x<-1 and x>1. There's no way it could: the training data doesn't cover these regions.

~~~~


