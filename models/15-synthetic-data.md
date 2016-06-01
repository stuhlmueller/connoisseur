---
layout: model
title: Synthetic Data for Learning to Infer Utility
---

Our data will come in episodes. Each episode consists of multiple steps. Each step contains an action, which affects the state. We'll start with a trivial example for the state and the transition functions.

~~~~
var sampleInitialState = function() {
  return randomInteger(5);
};

var sampleAction = function(state) {
  return 'plusone';
};

var transition = function(state, action) {
  if (action === 'plusone') {
    return state + 1;
  } else {
    print("error: unknown action");
  }
};

var state = sampleInitialState();
var action = sampleAction(state);
var nextState = transition(state, action);

print({state: state, action: action, nextState: nextState});
~~~~

Our basic data-generating process then looks like this:

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

var sampleInitialState = function() {
  return randomInteger(5);
};

var sampleAction = function(state) {
  return 'plusone';
};

var transition = function(state, action) {
  if (action === 'plusone') {
    return state + 1;
  } else {
    print("error: unknown action");
  }
};
///

var sampleStep = function(state) {
  var action = sampleAction(state);
  var newState = transition(state, action);
  var step = {
    action: action,
    stateAfterAction: newState      
  };
  return step;
};

var sampleSteps = function(numSteps, stepsSoFar) {
  if (numSteps === 0) {
    return stepsSoFar;
  } else {
    var prevState = last(stepsSoFar).stateAfterAction;
    var step = sampleStep(prevState);
    var steps = stepsSoFar.concat([step]);
    return sampleSteps(numSteps - 1, steps);
  }
};


var sampleEpisode = function(numSteps, episodeIndex) {
  var initialStep = {
    action: 'initialize',
    stateAfterAction: sampleInitialState()    
  };
  var steps = sampleSteps(numSteps, [initialStep]);
  return {
    steps: steps,
    index: episodeIndex
  };
};

var sampleEpisodes = function(options) {
  var numEpisodes = options.numEpisodes || 1;
  var stepsPerEpisode = options.stepsPerEpisode || 5;
  var episodes = repeatIndexed(numEpisodes, function(i){
    return sampleEpisode(stepsPerEpisode, i);
  });
  return episodes;
};


var printEpisode = function(episode) {
  print("Episode " + episode.index + ":");
  map(function(step){print(step);}, episode.steps);
  return;
};

var printEpisodes = function(episodes) {
  map(printEpisode, episodes);
  return;
};


var episodes = sampleEpisodes({
  numEpisodes: 3,
  stepsPerEpisode: 4
});

printEpisodes(episodes);
~~~~

We'll want to use this data-generating process for different priors on states and different transition functions.
Currently, these parameters are global variables. Let's refactor the above a bit so that they are more clearly parameters:

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

var printEpisodes = function(episodes) {
  map(printEpisode, episodes);
  return;
};
///

var makeDataGenerator = function(options) {

  var sampleAction = options.sampleAction;
  var transition = options.transition;
  var sampleInitialState = options.sampleInitialState;

  var sampleStep = function(state) {
    var action = sampleAction(state);
    var newState = transition(state, action);
    var step = {
      action: action,
      stateAfterAction: newState      
    };
    return step;
  };

  var sampleSteps = function(numSteps, stepsSoFar) {
    if (numSteps === 0) {
      return stepsSoFar;
    } else {
      var prevState = last(stepsSoFar).stateAfterAction;
      var step = sampleStep(prevState);
      var steps = stepsSoFar.concat([step]);
      return sampleSteps(numSteps - 1, steps);
    }
  };

  var sampleEpisode = function(numSteps, episodeIndex) {
    var initialStep = {
      action: 'initialize',
      stateAfterAction: sampleInitialState()    
    };
    var steps = sampleSteps(numSteps, [initialStep]);
    return {
      steps: steps,
      index: episodeIndex
    };
  };

  var sampleEpisodes = function(options) {
    var numEpisodes = options.numEpisodes || 1;
    var stepsPerEpisode = options.stepsPerEpisode || 5;
    var episodes = repeatIndexed(numEpisodes, function(i){
      return sampleEpisode(stepsPerEpisode, i);
    });
    return episodes;
  };

  return sampleEpisodes;

};


var sampleEpisodes = makeDataGenerator({
  sampleInitialState: function() {
    return randomInteger(5);
  },
  sampleAction: function(state) {
    return 'plusone';
  },
  transition: function(state, action) {
    if (action === 'plusone') {
      return state + 1;
    } else {
      print("error: unknown action");
    }
  }
});

var episodes = sampleEpisodes({
  numEpisodes: 3,
  stepsPerEpisode: 4
});

printEpisodes(episodes);
~~~~

We'd like to learn how good each action is with respect to the utility function of some agent. We'll assume that this utility function applies to the state, and actions are good to the extent that they improve the state's utility. (The real situation is more complicated and we may have to take into account whether the state enables future high-quality actions, but we can assume that this is folded into the agent's utility function for now.)

Let's extend our example from above:

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

var printEpisodes = function(episodes) {
  map(printEpisode, episodes);
  return;
};
///

var makeDataGenerator = function(options) {

  var sampleAction = options.sampleAction;
  var transition = options.transition;
  var sampleInitialState = options.sampleInitialState;
  var utility = options.utility;
  
  var sampleStep = function(state) {
    var action = sampleAction(state);
    var newState = transition(state, action);
    var step = {
      action: action,
      stateAfterAction: newState,
      actionUtility: utility(newState) - utility(state)
    };
    return step;
  };

  var sampleSteps = function(numSteps, stepsSoFar) {
    if (numSteps === 0) {
      return stepsSoFar;
    } else {
      var prevState = last(stepsSoFar).stateAfterAction;
      var step = sampleStep(prevState);
      var steps = stepsSoFar.concat([step]);
      return sampleSteps(numSteps - 1, steps);
    }
  };

  var sampleEpisode = function(numSteps, episodeIndex) {
    var initialStep = {
      action: 'initialize',
      stateAfterAction: sampleInitialState()    
    };
    var steps = sampleSteps(numSteps, [initialStep]);
    return {
      steps: steps,
      index: episodeIndex
    };
  };

  var sampleEpisodes = function(options) {
    var numEpisodes = options.numEpisodes || 1;
    var stepsPerEpisode = options.stepsPerEpisode || 5;
    var episodes = repeatIndexed(numEpisodes, function(i){
      return sampleEpisode(stepsPerEpisode, i);
    });
    return episodes;
  };

  return sampleEpisodes;

};


var sampleEpisodes = makeDataGenerator({
  sampleInitialState: function() {
    return randomInteger(5);
  },
  sampleAction: function(state) {
    return flip(.5) ? 'plusone' : 'minusone';
  },
  transition: function(state, action) {
    if (action === 'plusone') {
      return state + 1;
    } else if (action === 'minusone') {
      return state - 1;
    } else {
      print("error: unknown action");
    }
  },
  utility: function(state) {
    return state;
  }
});

var episodes = sampleEpisodes({
  numEpisodes: 3,
  stepsPerEpisode: 4
});

printEpisodes(episodes);
~~~~

On future pages:

- Signals (about goodness of actions)
- More complex state
- Distinguishing observable and unobservable state
- Sharing global state across episodes
- Query functions for active learning
