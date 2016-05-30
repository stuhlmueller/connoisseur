---
layout: model
title: Outline of generative models
author: Owain
---


# Outline of different possible generative models

## Notation

There are fixed functions u, b and g. There's a sequence of context observations {x_i}. We also observe the outputs of these functions for each context observations. The training data is typically of form `{(x_i, g(x_i), b(x_i))}` and the test data of form `{(x_i, b(x_i))}`, with the goal to predict `u(x_i)`. The variables are as follows: 

- `x_i` is the i-th observation of the context variable

- u is the utility function on the context x. We write u_i for `u(x_i)`, i.e. the utility of the i-th context.

- b is a function that outputs the "biased" side-information. It's either a function of `x_i` or `u(x_i)`. We write b_i for the i-th observation of biased side-information.

- g is a function that outputs the "gold-standard" side-information. It's either a function of `x_i` or `u(x_i)`. We write g_i for the i-th observation of the gold-standard side-information.



## Model 1: Utility observed, biased side-info depends directly on utility
This is the simplest model of learning from side information. There is a utility function `u` and a "biased side-information" function `b`. In this case, `b` acts directly on `u(x)` (e.g. by adding noise) rather than on the context `x`. There is no "gold-standard" side-information. Depending on the form of function b, this is similar to the "supervised learning with noisy labels" setting. 

###Bayes net
Note: we use plate notation for the series of observations of `x_i` and functions of it. While `u(x_i)` depends on x_i, we will generally leave out the dependency since we won't infer anything about the marginal distribution P(x_i). 


###Concrete example:
x ~ N(0,1)
u(x) = a(x), where a ~ N(0,1)
b_i = b(u_i) = u_i + constant + e_b, where constant ~ N(0,1) and e_b ~ Gamma(1,1)

**Training**: (x_i, u(x_i), b(u(x_i)) )
**Test**: (x_i, b(u(x_i) ))  ->  `u(x_i)`

###Comments on P( `u(x_i)` | b_i, x_i, training):
How does b_i (the biased side-information for observation x_i) help predict `u(x_i)`? Suppose we are fairly uncertain about `u(x_i)` because `x_i` is outside the training data. We might have learned that b has low noise and low bias. And so `u(x_i)` is likely close to b_i. 

###Tractability
For exact inference, we'd be inferring both u and b. Inference over u might be hard. Inference over b (if we knew u) should be easy because it's a function from R to R and the number of data points N is small.

One crude shortcut is to learn b from the training data and not bother learning u. Then just use the prior on u to infer `u(x_i)` given b(`u(x_i)`). If b does not degrade `u(x_i)` too much, then we'll make reasonable estimates. 

## Model 2: Utility observed, biased side-info depends on context
If the biased side-info depends directly on the utility, then we can't represent side-info that always ignores a particular component of x_i. So in a more flexible generative model for side-information `b` acts on `x_i` directly rather than `u(x_i)`.

We need a prior on `u` and `b` that makes them dependent. This is generally done by parameterizing the functions and having them share some parameters. One option is to have a single matrix or neural network from vector `x` to vector `(u(x), b(x))` and have a prior that promotes similar functions for each output. 

### Concrete example
x ~ N(0,1)
u(x) = a(x), where a ~ N(0,1)
b_i = b(x_i) = (a+b)x + e, where b ~ N(0,1), e ~ Gamma(1,1)

**Training**: (x_i, u(x_i), b(x_i) )
**Test**: (x_i, b(x_i))  ->  u(x_i)

###Comments on P( u(x_i) | b_i, x_i, training):
How does b_i (the biased side-information for observation x_i) help predict `u(x_i)`? In Model 2, b(x_i) gives information about b, which gives information about u (since they are dependent functions). If we already know the function `b`, then learning `b(x_i)` provides us no new information about `u(x_i)`. By contrast, in Model 1 above, `b_i` can provide information about `u(x_i)` even if we already know `b`.


###Tractability
The obvious approach is to choose a prior over matrices or neural nets from x to (u(x), b(x)) that facilitates tractable inference. At test time, we need to be able to condition the matrix/net on a value (x,b(x)), i.e. with u(x) missing from the output vector. (I'd guess there are standard techniques for dealing with this). 

-----------

## Model 3: Utility not observed, gold-standard depends directly on utility, biased side-info depends on context

In this case the utility is not observed directly. Instead we observe the "gold-standard" side-information, which is a variable that is a noisy function of `u(x)` for all `x`. Why have a gold-standard? If we have a broad prior on `u` and there is no gold-standard, then we won't be able to identify `u`. For example, we can't recover u(x) if a side-information variable is some arbitrary smooth deterministic function s(u(x)). If we're only interesting in learning `u` up to affine or monotone transformation, then the identifiability conditions will be different. 

In practice, we won't be able to learn much about `u` from a small training set if the gold-standard is a very noisy function of `u(x)`.

### Concrete example
x ~ N(0,1)
u(x_i) = a(x_i), where a ~ N(0,1)
g_i = g(u(x_i)) = u(x_i) + N(0,e), where e ~ Gamma(1,1)
b_i = b(x_i) = (a+b)x + e, where b ~ N(0,1), e ~ Gamma(1,1)

**Training**: (x_i, g(u(x_i)), b(x_i) )
**Test**: (x_i, b(x_i))  ->  u(x_i)

###Comments on P( u(x_i) | b_i, x_i, training):
This is very similar to the case of Model 2 above. The value b_i = b(x_i) tells us about `u(x_i)` only indirectly by telling us more about the function `b` which tells us about `u`. The only difference with Model 2 is that our training didn't provide direct observations of `u(x_i)` but only noisy ones via the gold-standard side-info. 


###Tractability
Again this is close to Model 2. Suppose we are learning a matrix/net for `u` and `b`. We now want the matrix/net to output (u(x),g(u(x)),b(x)) given input x. If g simply adds some zero-mean noise to u(x), this might be easy. (The general issue is how to constrain the spaces of matrices/nets to capture this constraint on the output variables). 


## Model 4: Utility not observed, all side-info is generated by functions similar to utility

## Model 5: Learn conditional distribution directly
Marginalize out some variables and reverse some arrows. You are learning 






