---
layout: model
title: Learning to Interpret Evidence using Regression 
---

Reliability as used in the previous section is the special case of learning to interpret evidence where the relation between the latent quantity and the evidence is the identity function with symmetric noise.

Now we'll assume that there is some (for now linear) function that relates different kinds of evidence to a latent quality we care about in a given context. We won't be able to exactly represent this function, so we model it as a stochastic function. For simplicity, we'll start by assuming that we can directly observe the latent quality at training time. As a consequence, the following will essentially be a regression problem.

For example, our context could be a set of features of a movie, our latent quality could be the user's enjoyment, and our evidence could be other users' judgments.

Our training data looks like this:

~~~~
var generateDatum = function() {
  var features = repeat(5, function(){return flip() ? 1 : 0;});
  var ratings = [
    features[0] * features[1],
    features[2] + 2 * features[3],
    features[2] * features[4]
  ];
  var judgment = ((2 * features[0] + features[1] - 2 * features[2] - features[3]) * (features[4] + 1));
  return {
    features: [features[0], features[1], features[2]],
    ratings: ratings,
    judgment: [judgment]
  }
};
  
repeat(10, function(){
  print(generateDatum());
})

var trainingData = repeat(20, generateDatum);
var testData = repeat(20, generateDatum);

wpEditor.put('trainingData', trainingData);
wpEditor.put('testData', trainingData);
~~~~

At test time, we will supply data of the same form, but with judgments omitted.

~~~~
///fold:
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

var trainingData = wpEditor.get('trainingData');
var testData = wpEditor.get('testData');
///

var errorOnData = function(f, data) {
  var error = sum(map(function(datum){
    var input = datum.features.concat(datum.ratings);
    return squaredError(f(input), Vector(datum.judgment));
  }, trainingData));
  return error;
};

var model = function() {
  
  var f = functionPrior([6, 1]);
  
  // Condition on data
  var trainError = errorOnData(f, trainingData);
  factor(-trainError);
  
  // Test
  var testError = errorOnData(f, testData) / testData.length;
  console.log(testError.x);
  return f;
};

var params = Optimize(model, {
  steps: 1000,
  method: {
    gd: {stepSize: 0.001}
  },
  estimator: {ELBO: {samples: 5}}});

var finalPredictor = SampleGuide(model, {params: params, samples: 1}).support()[0];

map(function(datum){
  var prediction = finalPredictor(datum.features.concat(datum.ratings)).data[0];
  print(_.assign(datum, {prediction: prediction}));
}, testData)
~~~~

This works well---unsurprisingly, since our data-generating function is mostly linear and we're using a form of regression.

Extension:

- maintain uncertainty over functions (by using HMC, or richer variational family)
- make quality latent, observe only user's report that depends on it (this will increase our uncertainty over functions, and so we'd also want to follow a more Bayesian approach here)
- maybe contrast with approach where we learn a generative model (from context to ratings)