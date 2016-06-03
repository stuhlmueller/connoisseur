---
layout: model
title: Evaluating a Regression Predictor on Simple Markovian Actions
---

We will now apply the evaluation framework introduced in the previous section to the simple sequential state space and regression predictor that we saw earlier. Let's first load the key framework functions back into our workspace:

~~~~
///fold:
var repeatIndexed = function(n, fn) {
  var helper = function(m, offset) {
    if (m == 0) {
      return [];
    } else if (m == 1) {
      return [fn(offset)]; // Pass the offset to fn, this is the difference with the built-in repeat
    } else {
      var m1 = Math.ceil(m / 2),
          m2 = m - m1;
      return helper(m1, offset).concat(helper(m2, offset + m1));
    }
  }

  return helper(n, 0);
};

var last = function(xs) {
  return xs[xs.length - 1];
};

var printEpisode = function(episode) {
  print("Episode " + episode.index + ":");
  map(function(step){print(step);}, episode.steps);
  return;
};

var printEpisodes = function(data) {
  print('globalState: ' + JSON.stringify(data.globalState));
  map(printEpisode, data.episodes);
  return;
};

var makeDataGenerator = function(options) {

  var initialAction = options.initialAction; // {author: 'system', value: 'initialize'}
  var sampleAction = options.sampleAction;
  var transition = options.transition;
  var sampleEpisodeState = options.sampleEpisodeState;
  var sampleGlobalState = options.sampleGlobalState;  
  var utility = options.utility;
  
  var sampleStep = function(globalState, episodeState) {
    var action = sampleAction(globalState, episodeState);
    var newEpisodeState = transition(globalState, episodeState, action);
    var actionUtility = utility(globalState, episodeState, action, newEpisodeState);
    var step = {
      action: action,
      stateAfterAction: newEpisodeState,
      actionUtility: actionUtility
    };
    return step;
  };

  var sampleSteps = function(globalState, numSteps, stepsSoFar) {
    if (numSteps === 0) {
      return stepsSoFar;
    } else {
      var prevState = last(stepsSoFar).stateAfterAction;
      var step = sampleStep(globalState, prevState);
      var steps = stepsSoFar.concat([step]);
      return sampleSteps(globalState, numSteps - 1, steps);
    }
  };

  var sampleEpisode = function(globalState, numSteps, episodeIndex) {
    var initialStep = {
      action: initialAction,
      stateAfterAction: sampleEpisodeState(globalState),
      actionUtility: 0
    };
    var steps = sampleSteps(globalState, numSteps, [initialStep]);
    return {
      steps: steps,
      index: episodeIndex
    };
  };

  var sampleEpisodes = function(options) {
    var numEpisodes = options.numEpisodes || 1;
    var stepsPerEpisode = options.stepsPerEpisode || 5;
    var globalState = sampleGlobalState();
    var episodes = repeatIndexed(numEpisodes, function(i){
      return sampleEpisode(globalState, stepsPerEpisode, i);
    });
    return {
      globalState: globalState,
      episodes: episodes
    };
  };

  return sampleEpisodes;

};


var makeDataPreparer = function(options) {

  var prepareTrainingStep = options.prepareTrainingStep;
  var prepareTestStep = options.prepareTestStep;  

  var prepareTrainingEpisode = function(episode) {
    return { steps: map(prepareTrainingStep, episode.steps) };
  };  

  var prepareTestEpisode = function(episode) {
    return { steps: map(prepareTestStep, episode.steps) };
  };

  var stepToUtility = function(step) {
    return step.actionUtility;
  };

  var episodeToUtilities = function(episode) {
    return map(stepToUtility, episode.steps);
  };

  var prepareData = function(data) {
    var i = Math.floor(data.episodes.length / 2);
    var trainingEpisodes = data.episodes.slice(0, i);
    var testEpisodes = data.episodes.slice(i, data.episodes.length);
    return {
      original: data,
      training: {
        episodes: map(prepareTrainingEpisode, trainingEpisodes)
      },
      test: {
        episodes: map(prepareTestEpisode, testEpisodes),
        utilities: map(episodeToUtilities, testEpisodes)
      }
    };
  };

  return prepareData;
};

var makeEvaluator = function(options) {
  
  var generateData = options.dataGenerator;
  var prepareData = options.dataPreparer;
  var train = options.predictor.train;
  var predict = options.predictor.predict;
  
  var getScore = function(predictorState, testData, utilities) {
    // FIXME: utilities only matter up to affine transformation
    var predictions = predict(predictorState, testData);
    return sum(map2(
      function(episodeUtilities, episodePredictions){
        return sum(map2(
          function(utility, prediction){
            // FIXME: need to change this for continuous distributions
            return prediction.score(utility);
          },
          episodeUtilities, episodePredictions));
      },
      utilities, predictions));    
  };
  
  var evaluate = function(options) {
    var scores = repeat(options.numRepeats, function() {
      print('Sampling data...');
      var rawData = generateData(options);
      print('Preparing data...');
      var data = prepareData(rawData);
      print('Training predictor...');
      var predictorState = train(data.training.episodes);
      print('Testing predictor...');
      var score = getScore(predictorState, data.test.episodes, data.test.utilities);
      print('Done. Score: ' + score);
      return score;
    });
    return scores;
  };

  return evaluate;  
};
///

wpEditor.put('printEpisodes', printEpisodes);
wpEditor.put('makeDataGenerator', makeDataGenerator);
wpEditor.put('makeDataPreparer', makeDataPreparer);
wpEditor.put('makeEvaluator', makeEvaluator);
~~~~

Now we'll define the data generator, preparer, predictor, and evaluator for our example. Let's start with the data generator:

~~~~
var printEpisodes = wpEditor.get('printEpisodes');
var makeDataGenerator = wpEditor.get('makeDataGenerator');

var dataGenerator = makeDataGenerator({
  sampleGlobalState: function() {
    return {
      aliceIsHelpful: flip(.5),
      bobIsHelpful: flip(.5)
    };
  },
  sampleEpisodeState: function(globalState) {
    return randomInteger(5);
  },
  sampleAction: function(globalState, episodeState) {
    var author = flip(.5) ? 'alice' : 'bob';
    var isHelpful = globalState[author + 'IsHelpful'];
    var p_help = isHelpful ? .8 : .2;
    var value = flip(p_help) ? 'plusone' : 'minusone';
    return {
      author: author,
      value: value
    };
  },
  initialAction: {
    author: 'system',
    value: 'initialize'
  },
  transition: function(globalState, episodeState, action) {
    var value = action.value;
    if (value === 'plusone') {
      return episodeState + 1;
    } else if (value === 'minusone') {
      return episodeState - 1;
    } else {
      print("error: unknown action");
    }
  },
  utility: function(globalState, episodeState, action, newEpisodeState) {
    return episodeState;
  }
});

var exampleData = dataGenerator({
  numEpisodes: 3,
  stepsPerEpisode: 5
});

wpEditor.put('dataGenerator', dataGenerator);
wpEditor.put('exampleData', exampleData);

printEpisodes(exampleData);
~~~~

The data preparer only makes the author for each action available to the predictor. We'll also observe utility during training:

~~~~
var makeDataPreparer = wpEditor.get('makeDataPreparer');
var exampleData = wpEditor.get('exampleData');

var dataPreparer = makeDataPreparer({
  prepareTrainingStep: function(step) {
    return { author: step.action.author, utility: step.actionUtility };
  },
  prepareTestStep: function(step) {
    return { author: step.action.author }
  }
});

wpEditor.put('dataPreparer', dataPreparer);

var preparedData = dataPreparer(exampleData);

print("Training:")
print(JSON.stringify(preparedData.training, null, 2));

print("Test:")
print(JSON.stringify(preparedData.test, null, 2));

wpEditor.put('preparedData', preparedData);
~~~~

The predictor:

~~~~
var makeRegressionPredictor = function(options) {

  var stepToInput = options.stepToInput;
  var stepToOutput = options.stepToOutput;

  var encodeStep = function(step) {
    return {
      input: stepToInput(step),
      output: stepToOutput(step)
    };
  };

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

  var matrixToFunction = function(matrix) {
    return function(x) {
      var u = sample(Uniform({a: 0, b: 1}), { guide: Uniform({a: 0, b: 1}) });
      var a = T.concat(x, Vector([u, 1]));
      var b = T.reshape(a, [x.length+2, 1]);
      var input = T.transpose(b);
      return T.dot(input, matrix);
    };
  }

  var functionPrior = function(dims) {
    var matrixDims = [dims[0] + 2, dims[1]];
    var matrix = sampleGaussianMatrix(matrixDims, 0, 1, 0);
    var f = matrixToFunction(matrix);
    _.assign(f, {'matrix': matrix});
    return f;
  };

  var squaredError = function(m1, m2) {
    var x = T.add(m1, T.mul(m2, -1));
    return T.sumreduce(T.mul(x, x));
  };

  var model = function(episodes) {
    var f = functionPrior([3, 1]);
    var score = sum(map(function(episode){
      return sum(map(function(step){
        var stepDatum = encodeStep(step);
        return squaredError(f(stepDatum.input), stepDatum.output);
      }, episode.steps));
    }, episodes));  
    factor(-score);
    return f.matrix;
  };  

  var train = function(episodes, opts) {
    var options = {
      steps: (opts && opts.steps) ? opts.steps : 500,
      stepSize: (opts && opts.stepSize) ? opts.stepSize : 0.01
    };
    var trainModel = function(){return model(episodes);};
    var params = Optimize(trainModel, {
      steps: options.steps,
      method: {
        gd: {stepSize: options.stepSize}
      },
      estimator: {ELBO: {samples: 20}}});
    var trainDist = SampleGuide(trainModel, {params: params, samples: 1});
    var trainedMatrix = trainDist.support()[0]
    return trainedMatrix;
  };

  var predict = function(trainedMatrix, episodes) {
    var f = matrixToFunction(trainedMatrix);
    return map(function(episode){
      return map(function(step){
        var input = stepToInput(step);
        var predictDist = Infer({method: 'rejection', samples: 1000}, function() {
          return f(input).data[0];
        });
        // return predictDist;
        return expectation(predictDist);
      }, episode.steps);
    }, episodes);
  };

  return {
    train: train,
    predict: predict
  };
};

var authorEncoding = {
  alice: [1, 0, 0],
  bob: [0, 1, 0],
  system: [0, 0, 1]
};


var regressionPredictor = makeRegressionPredictor({
  stepToInput: function(step) {
    return Vector(authorEncoding[step.author]);
  },
  stepToOutput: function(step) {
    return Vector([step.utility]);
  }
});

wpEditor.put('makeRegressionPredictor', makeRegressionPredictor);
wpEditor.put('regressionPredictor', regressionPredictor);

var train = regressionPredictor.train;
var predict = regressionPredictor.predict;

var trainedState = train([
  {
    "steps": [
      {
        "author": "bob",
        "utility": -1
      },
      {
        "author": "bob",
        "utility": -1
      },
      {
        "author": "alice",
        "utility": 1
      },
      {
        "author": "bob",
        "utility": 1
      }
    ]
  }
]);

predict(trainedState, [
  {
    "steps": [
      {
        "author": "alice"
      },
      {
        "author": "bob"
      }
    ]
  }
]);
~~~~

Finally, the evaluator itself:

~~~~
///fold:
var makeEvaluator = function(options) {
  
  var generateData = options.dataGenerator;
  var prepareData = options.dataPreparer;
  var train = options.predictor.train;
  var predict = options.predictor.predict;
  
  var getScore = function(predictorState, testData, utilities) {
    // FIXME: utilities only matter up to affine transformation
    var predictions = predict(predictorState, testData);
    return sum(map2(
      function(episodeUtilities, episodePredictions){
        return sum(map2(
          function(utility, prediction){
            if (prediction.score !== undefined) {
              return prediction.score(utility);
            } else {
              // FIXME
              return Gaussian({mu: prediction, sigma: 1}).score(utility);
            }
          },
          episodeUtilities, episodePredictions));
      },
      utilities, predictions));    
  };
  
  var evaluate = function(options) {
    var scores = repeat(options.numRepeats, function() {
      print('Sampling data...');
      var rawData = generateData(options);
      print('Preparing data...');
      var data = prepareData(rawData);
      print('Training predictor...');
      var predictorState = train(data.training.episodes);
      print('Testing predictor...');
      var score = getScore(predictorState, data.test.episodes, data.test.utilities);
      print('Done. Score: ' + score);
      return score;
    });
    return scores;
  };

  return evaluate;  
};

wpEditor.put('makeEvaluator', makeEvaluator);
///

var evaluator = makeEvaluator({
  dataGenerator: wpEditor.get('dataGenerator'),
  dataPreparer: wpEditor.get('dataPreparer'),
  predictor: wpEditor.get('regressionPredictor')
});

evaluator({
  numEpisodes: 2,
  stepsPerEpisode: 5,
  numRepeats: 3
});
~~~~

Next steps:

1. Provide the option to print out a lot more information, figures; it should be easy to get an intuitive sense for what a predictor is doing, and how well it's doing
2. Use a more principled scoring function for evaluating actual utilities vs predicted (continuous) distributions
3. Apply to Owain's example(s)
4. Write a version of the regression predictor that handles uncertainty correctly: Due to the use of variational inference, and the selection of a single matrix/function above, we're also not correctly capturing uncertainty in our distribution over functions when we're doing prediction. We could address this by (1) computing a distribution on matrices using HMC (with independent random variable for each matrix element) and (2) storing this distribution so that we can use it for prediction.

Other things to explore:

- Supply non-uniform source of noise for the induced stochastic function
- Make the actual data-generating process more complex, make prior over functions more expressive (use Bayesian neural nets), make state representation more sophisticated
- Don't observe utility directly, only observe a "gold standard" signal some of the time
- Observe weaker signals
- Contrast generative models (that try to model the joint distribution on state and utility) with the supervised approach (where we try to learn a stochastic function from state to utility)
- Evaluate how this approach scales to larger datasets