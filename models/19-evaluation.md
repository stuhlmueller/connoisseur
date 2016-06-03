---
layout: model
title: Evaluating Algorithms that Learn to Infer Utility
---

We're now going to write a more structured framework for evaluating utility learners on different problems.

Our data generating process is going to be the same as previously (generating multiple episodes, each with multiple steps, with fixed by unknown global state and mutable per-episode state). A trivial example:

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

wpEditor.put('makeDataGenerator', makeDataGenerator);

///

var trivialDataGenerator = makeDataGenerator({
  sampleGlobalState: function() { return {}; },
  sampleEpisodeState: function(globalState) { return false; },
  sampleAction: function(globalState, episodeState) { return flip(.5); },
  initialAction: false,
  transition: function(globalState, episodeState, action) { return action; },
  utility: function(globalState, episodeState, action, newEpisodeState) { return newEpisodeState ? 1 : 0; }    
});

var data = trivialDataGenerator({
  numEpisodes: 2,
  stepsPerEpisode: 5
});

wpEditor.put('trivialDataGenerator', trivialDataGenerator);
wpEditor.put('data', data);

printEpisodes(data);
~~~~

A predictor is trained on episodes (such as the ones above) and learns to predict utility for new episodes. It may or may not be given explicit utilities for the training episodes. It jointly makes a prediction for all steps in all test episodes.

~~~~
var data = wpEditor.get('data');

var makeTrivialPredictor = function() {
  var train = function(episodes) {
    return null;
  };
  var predict = function(trainedState, episodes) {
    return map(
      function(episode){
        return map(
          function(step){
            return Enumerate(function(){
              return flip(step.action ? .7 : .3) ? 1 : 0;
            });
          }, 
          episode.steps);
      },
      episodes)
  }
  return {
    train: train,
    predict: predict
  }
};

var trivialPredictor = makeTrivialPredictor();

wpEditor.put('trivialPredictor', trivialPredictor);

var predict = trivialPredictor.predict;
predict(null, data.episodes);
~~~~

Our predictors won't directly observe all of the data---they will in general only observe a subset of the data (e.g. not including true utilities, only various correlated signals). We also need to split the data into training and test data. We do both of these tasks using a data preparer:

~~~~
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

var trivialDataPreparer = makeDataPreparer({
  prepareTrainingStep: function(step) {
    return { action: step.action, utility: step.actionUtility };
  },
  prepareTestStep: function(step) {
    return { action: step.action }
  }
});

wpEditor.put('makeDataPreparer', makeDataPreparer);
wpEditor.put('trivialDataPreparer', trivialDataPreparer);

var data = wpEditor.get('data');
var prepared = trivialDataPreparer(data);

print("Training:")
print(prepared.training);

print("Test:")
print(prepared.test);

wpEditor.put('trainingData', prepared.training);
wpEditor.put('testData', prepared.test);
~~~~

So, now we have:

- A way to generate data
- A way to select what data to show to the predictor
- A predictor

Now we would like to evaluate how well the predictor does on this dataset in terms of recovering utility for unseen states.

~~~~
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

var trivialEvaluator = makeEvaluator({
  dataGenerator: wpEditor.get('trivialDataGenerator'),
  dataPreparer: wpEditor.get('trivialDataPreparer'),
  predictor: wpEditor.get('trivialPredictor')
});

trivialEvaluator({
  numEpisodes: 2,
  stepsPerEpisode: 5,
  numRepeats: 3
});
~~~~

Next steps:

- Apply this to my regression example
  - This requires a change to the scoring procedure
  - The predictor needs to include the data embedding procedure
- Apply this to Owain's example(s)