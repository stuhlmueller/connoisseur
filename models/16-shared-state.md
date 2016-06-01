---
layout: model
title: Sharing Global State Across Episodes
---

As a first extension of our data generator, we'll add global state that is shared across episodes. The motivation for this is that some latent variables, such as whether a particular agent tends to be helpful or not, are fairly stable and are among the most helpful pieces of information for predicting whether an action has high utility or not.

~~~~
~~~~