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

To add:

- metadata (who initiated action)
- utilities
- signals
- global state
- query functions for active learning



~~~~
var model = function() {
  var action = uniformDraw(actions);
  // ...
};

// - our data will consist of a sequence of actions
// - we're working the setting of string manipulation for now
//   (maybe we can be more general)
// - there is a target string that alice prefers
// - other users can make edits to the string
// - users can provide different kinds of signals
// - we can ask alice different kinds of questions, both global ones
//   ("how good is this string?") and local ones 


// our data consists of episodes
// each episode contains a sequence of actions
// we'll also have metadata associated with actions
// - who authored it
// - who liked/disliked it
//   - as a special case, whether alice likes it
// we're also able to acquire expensive signals at the end

// intuitions:
// - if a contribution is reverted by a highly rated user, it was probably bad
// - if a deletion is rated highly, the contribution can't have been
//   good; and vice versa

// how will users differ:
// - some will be more noisy than others
// - some will only be good at particular kinds of actions
//   - some will be good at rating, but not at direct contributions,
//     and vice versa
//   - some will only be good wrt certain characters
// - some will be good at rating others' contributions, but won't be
//   good at rating their own contributions
// - some will be biased when rating particular kinds of contributions,
//   or particular kinds of users

// More prerequisites:
// - signals/judgments of actions
// - biased/noise actions

// var generateDatum = {
//   ... 
// };

var data = [
  {
    sequence: [
      {
        observableState: "abc",
        hiddenState: "def",
        action: "",
        author: ""
      },
      {},
      {}],    
  }  
];
~~~~

This generator will also supply a function that can be queried in an active learning setting.