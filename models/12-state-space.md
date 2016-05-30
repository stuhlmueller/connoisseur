---
layout: model
title: A Simple State Space for Sequential Actions
---

In the previous section, we already generated data as basis for a simple regression model. Before we move to more complex models, I'll write a more complete generator for the kind of data that we're interested in. To do so, I'll first define a simple state space for sequential actions, namely the space of lists of characters, with three functions (`push`, `pop`, `swap`) for manipulating elements in this space.

Let's first define our three actions:

~~~~
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

wpEditor.put('moveFunctions', moveFunctions);

// Insert a letter at the end
var push = moveFunctions.push;
print(push(["a","b","c"], "d"));

// Remove first letter
var pop = moveFunctions.pop;
print(pop(["a", "b", "c"]));

// Swap two letters
var swap = moveFunctions.swap;
print(swap(["a", "b", "c"], 0));
~~~~

We can now define a distribution on an action, given a state:

~~~~
var moveFunctions = wpEditor.get('moveFunctions');

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

wpEditor.put('sampleMove', sampleMove);

sampleMove(["a", "b"]);
~~~~

This naturally generalizes to a prior on a sequence of actions:

~~~~
var sampleMove = wpEditor.get('sampleMove');
var moveFunctions = wpEditor.get('moveFunctions');

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

wpEditor.put('applyMove', applyMove);
wpEditor.put('applyMoves', applyMoves);
wpEditor.put('sampleMoves', sampleMoves);

sampleMoves(["a", "b"], 2);
~~~~

We'll pretty-print such sequences using the following functions:

~~~~
var sampleMoves = wpEditor.get('sampleMoves');
var applyMove = wpEditor.get('applyMove');

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

var initState = [];
var moves = sampleMoves(initState, 20);
showMoves(initState, moves);
~~~~