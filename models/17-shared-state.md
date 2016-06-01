---
layout: model
title: Sharing Global State Across Episodes
---

As a first extension of our data generator, we'll add global state that is shared across episodes. The motivation for this is that some latent variables, such as whether a particular agent tends to be helpful or not, are fairly stable and are among the most helpful pieces of information for predicting whether an action has high utility or not.

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
///

var makeDataGenerator = function(options) {

  var sampleAction = options.sampleAction;
  var transition = options.transition;
  var sampleEpisodeState = options.sampleEpisodeState;
  var sampleGlobalState = options.sampleGlobalState;  
  var utility = options.utility;
  
  var sampleStep = function(globalState, episodeState) {
    var action = sampleAction(globalState, episodeState);
    var newEpisodeState = transition(globalState, episodeState, action);
    var actionUtility = utility(globalState, newEpisodeState) - utility(globalState, episodeState);
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
      action: 'initialize',
      stateAfterAction: sampleEpisodeState(globalState)    
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
wpEditor.put('printEpisodes', printEpisodes);


var sampleEpisodes = makeDataGenerator({
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
  utility: function(globalState, episodeState) {
    return episodeState;
  }
});

var data = sampleEpisodes({
  numEpisodes: 3,
  stepsPerEpisode: 4
});

printEpisodes(data);
~~~~