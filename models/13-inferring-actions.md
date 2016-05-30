---
layout: model
title: Inferring Sequences of Actions
---

Let's see if we can infer actions that lead to a specific outcome.

~~~~
///fold:
var moveFunctions = {
  push: function(state, s) {
    return state.concat([s]);
  },
  pop: function(state) {
    return state.slice(1);
  },
  swap: function(state, i) {
    var a = state[i];
    var b = state[i + 1];
    return state.slice(0, i).concat([b, a]).concat(state.slice(i + 2));    
  }
};

var randomSymbol = function() {
  return uniformDraw(['a', 'b', 'c', 'd', 'e']);
};

var samplePush = function(state) {
  return {
    name: 'push',
    data: randomSymbol()
  }
};

var samplePop = function(state) {
  return {
    name: 'pop',
    data: null
  }
};

var sampleSwap = function(state) {
  return {
    name: 'swap',
    data: randomInteger(state.length - 1)
  }
};

var availableMoves = function(state) {
  if (state.length >= 2) {
    return [samplePush, samplePop, sampleSwap];
  } else if (state.length >= 1) {
    return [samplePush, samplePop];
  } else {
    return [samplePush];
  }
};

var sampleMove = function(state) {
  var sampler = uniformDraw(availableMoves(state));
  return sampler(state);
};

var applyMove = function(state, move) {
  var moveFunc = moveFunctions[move.name];
  return moveFunc(state, move.data);
};

var applyMoves = function(state, moves) {
  if (moves.length === 0) {
    return state;
  } else {
    return applyMoves(applyMove(state, moves[0]), moves.slice(1));
  }
};

var sampleMoves = function(state, n) {
  if (n === 0) {
    return [];
  } else {
    var move = sampleMove(state);
    var newState = applyMove(state, move);
    return [move].concat(sampleMoves(newState, n - 1));
  }
};

var showMove = function(state, move) {
  var nextState = applyMove(state, move);
  print(
    JSON.stringify(state) + 
    " --{" + move.name + 
    ((move.data !== null) ? " " + move.data : "") + "}--> " + 
    JSON.stringify(nextState));
}

var showMoves = function(state, moves) {
  if (moves.length === 0) {
    return;
  } else {    
    var move = moves[0];
    showMove(state, move);
    var nextState = applyMove(state, move);
    return showMoves(nextState, moves.slice(1));
  }
};

wpEditor.put('sampleMoves', sampleMoves);
wpEditor.put('applyMoves', applyMoves);
wpEditor.put('showMoves', showMoves);
///

var initState = [];
var targetState = ['a', 'b', 'c', 'd'];

var model = function() {
  var i = randomInteger(targetState.length * 2);
  var moves = sampleMoves(initState, i);
  var finalState = applyMoves(initState, moves);
  condition(_.isEqual(finalState, targetState));
  return moves;
}

var out = Infer({method: 'rejection', samples: 1}, model);

var moves = out.support()[0];

showMoves(initState, moves);
~~~~

This works, but gets slow very quickly as the state space grows. Maybe we can do better using MCMC? For this to work well, we'll need a soft comparison function that can tell us how far off our sequence of actions is from accomplishing the goal.

~~~~
var distance = dp.cache(function(s, t) {
  // This needs to be quick to compute...
  if ((s.length === 0) && (t.length === 0)) {
    return 0;
  } else if (s.length === 0) {
    return t.length;
  } else if (t.length === 0) {
    return s.length;
  } else if (s[0] === t[0]) {
    return distance(s.slice(1), t.slice(1));
  } else {
    return Math.min(
      distance(s, t.slice(1)) + 1,
      distance(s.slice(1), t) + 1
    );    
  }
});

wpEditor.put('distance', distance);

distance(['a', 'b', 'c', 'e', 'f'], ['b', 'c', 'd', 'f', 'e']);
~~~~

We can now infer somewhat longer sequences of actions:

~~~~
var sampleMoves = wpEditor.get('sampleMoves');
var applyMoves = wpEditor.get('applyMoves');
var showMoves = wpEditor.get('showMoves');
var distance = wpEditor.get('distance');


var initState = [];
var targetState = ['a', 'b', 'c', 'd', 'c', 'b'];

var model = function() {
  var i = randomInteger(targetState.length * 2);
  var moves = sampleMoves(initState, i);
  var finalState = applyMoves(initState, moves);
  factor(-10 * distance(finalState, targetState));
  return moves;
};

var out = Infer({method: 'MCMC', samples: 1, burn: 50000}, model);

var moves = out.support()[0];

showMoves(initState, moves);
~~~~