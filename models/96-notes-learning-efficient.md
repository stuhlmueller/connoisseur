---
layout: model
title: Notes on Learning preferences efficiently using side-information
author: Owain
---


I start by describing some Machine Learning problems concerned with using training data efficiently. I then motivate these problems in terms of having ML systems learn human preferences from data. Finally, I discuss some concrete applications.

**Sections:**

1. <a href="#intro">Introduction to learning with side-information and to bespoke feedback</a>

2. <a href="#motivation">Motivations from learning human preferences and values</a>

3. <a href="#concrete">Concrete problems in prediction and in RL.


<a id="intro"></a>

## 1. Machine Learning Problems

### Learning with side-information (i.e. heterogeneous training data)
One version of this problem was described by Christiano [here](https://medium.com/ai-control/semi-supervised-learning-from-side-information-483d5db474a2#.ifx6lua1w) and that post provides a good short introduction. In the setting of a prediction problem (as opposed to RL problem) where the goal is to predict target Y from input variable (or "context") X, learning from side-information is the following:

> Given an instance x and side information z', produce a prediction pred(y) to minimize Loss(y, pred(y)), where y is true target variable. The side information z' is a subset of all the z-variables (z1, z2, ... , zN).

So we have a standard prediction problem (predicting Y given X). But there is an additional variable Z, which has dimension N and components (z1, ... , zN). For any given instance x, we might have success to some of the "z-variables", i.e. the components of Z. We call the z-variables *side-information* because unlike X they are not always available (in particular they may be absent at test-time) and unlike Y they are not the target variable.

One setting for learning from side-information is *passive learning*. Here each z-variable, zi, is observed with probability pi for any given instance x. Another setting is *active learning*. This is the same as passive learning, except a cost C(x,zi) can be paid to observe z-variables that would otherwise be unobserved.

How do the z-variables relate to X and Y? We'll generally make some assumptions about the joint distribution on the variables X, Z (i.e. all the side information variables) and Y. One assumption is that the side-information z-variables are some transformation of Y (with noise added): i.e. zi = f(y) + noise. Another is that the z-variables are a function of X that is similar to the function related X to Y: i.e. zi = f1(x) and y = f2(x), and f1 and f2 are similar. 

Consider the following more concrete examples:

1. Suppose y is a human approval/utility rating for action x after extensive reflection. Then the z-variables could be approval judgments for lesser amounts of reflection. Or y might be a particular person's approval ratings and the z-variables might be the ratings of people similar to that person. This relates to Christiano's [post](https://medium.com/ai-control/turning-reflection-up-to-11-1bd6171afd21#.8m750yfqc) and various other posts on AI Control and elsewhere.

2. Suppose y is a carefully considered evaluation of a movie m by a person p. In this case, the variable x = (m,p), and so it's a standard user-movie recommendation setup. More specifically, y is p's rating for a movie after being paid to write a page-long review of the movie (after watching with no distractions). The z-variables might then be ratings of the movie by p after watching only the first hour and having lots of distractions. Alternatively, the z-variables might be ratings of movie m by other people (as in collaborative filtering). Note that Netflix actually employs people to review movies all day. Facebook uses a similar approach to provide scores for newsfeed items. In these cases, the "considered evaluation" is never observed for almost all users p but it can still be predicted from the side-information that is observed. In Facebook, the side-information is "Likes", commenting, click-throughs and so on. In Netflix, it is movie choices and movie ratings. See Slate article "Who Controls Your Newsfeed?" by Will Oremus. (Related examples: predicting expert code-review results from novice code-review results, predicting future citations from peer review scores). 

3. y is an expert transcription of noisy audio file x. Assume the speaker has a strong foreign or regional accent and that the topic is highly abstruse (e.g. math, professional sport or military tactics). The z-variables are transcriptions from random Mechanical Turk users (who are not provived with any tips or context). 

In these examples, the z-variables have the same dimension as y and a simple estimator uses the z's directly in place of y would do acceptably well at the prediction task. This relates to the Machine Learning literature on "Learning with Label Noise" (see survey by Frénay and Michel Verleysen). We could make additional assumptions about the form of the noise (as has been explored in the literature) such as:

> zi is independent of x given y. This would true in the case that zi is a function of y (and not of x directly). 

We could also consider settings where the z-variables are not noisy versions of y. For example, suppose the task is segmenting images or doing speech transcription. The z's could be partial versions of the transcription (since the human ran out of time or could only do some part of the task). Another possibility is that the z's are features of the x's. For instance, maybe the task is to label a situation (e.g. as just/unjust, safe/unsafe, normal/abnormal) where the situation is depicted via text or an image. Humans might be unsure of the label but can confidently pick out some relevant high-level features. 

Another variant is to consider an RL (or IRL) problem instead of a prediction problem. For the prediction problem, we allow Y to be any kind of variable (discrete, continuous, etc.) that depends on X. In the RL case, Y is the real-valued reward of the state X. The z-variables would then be noisy versions of Y that are more frequently observed by the agent or are cheaper to obtain via active learning. (This is just one kind of RL problem involving side-information. The side-information could also be policy advice instead of information about reward.)

--------

### Bespoke human feedback
In the example of side-information, we can think of the z-variables as signals about the target y that vary in *quality*. Formally, we could think of quality as a combination of mutual information and an easy-to-learn functional relationship between zi and y. One kind of high-quality signal is *bespoke* or *customized* feedback or training. By "bespoke feedback", we mean feedback that is customized to the specific algorithm that is being trained.

In standard supervised learning, an algorithm A is trained on examples {(x, y)}, where y is the prediction target for x and where y has no dependence on A. For example, in ImageNet, humans provide supervision/training by labelling images with object names. The labelling is carried out independently of the algorithms that are to be trained on the images. Researchers training the algorithm A might pre-process these training examples to make learning more tractable for A. But this usually doesn't change the relationship between the variables X and Y as encoded in the labels: the labels were produced by humans who saw the original X (before pre-processing) and did not see A.

With bespoke feedback the labels can be a function A. For example, the researchers labelling the images take the algorithm they are training into account. In the case of online learning, the researchers could also take into account A's history of guesses when labelling the next datapoint. To get an intuition for the distinction between regular training and bespoke feedback, think about learning a foreign language via a textbook vs. having a personal tutor (who personalizes each lesson). Christiano discusses bespoke feedback in the dialog setting [here](https://medium.com/ai-control/efficient-feedback-a347748b1557#.t2jpoz9f3) and the same problem for dialogs is discussed in this [survey](http://arxiv.org/abs/1512.05742) on page 6.

Some general motivations for bespoke feedback are:

1. The algorithm A will not be able to learn the true mapping from x to y perfectly but could learn some simplified mapping. This is similar to the motivation for pre-processing but the simplification of the training data occurs when the labels are generated and not afterwards. 

2. Algorithm A learns in a path-dependent manner and will learn the true mapping from x to y more quickly if it first learns a simplified mapping. (Optimal learners are not path dependent but good practical algorithms may be). 

The idea of *approval-directed training* for low-level actions is closely related to bespoke training. (For a particular approach to approval-directed training, see Christiano's post on [annotated functional programming](https://medium.com/ai-control/approval-directed-algorithm-learning-bf1f8fad42cd#.b8f6vpfjf). Earlier posts spell out the general approach). The approval-directed approach can be contrasted with a more standard ML approach to teaching a concept. On the standard approach, the concept of human preference would be conveyed by having humans label a set of situations (as they do in object recognition tasks). The problem is that these situations would be too abstract for humans to label with confidence (e.g. there is disagreement on topics like abortion, population ethics, etc.). The approval-directed approach is to instruct a *particular* algorithm in a *particular* background context. The hope is that giving training/feedback about very concrete actions is easier than providing abstract guidance. (Similarly: If you find yourself in an ambulance with critically injured friend and you are asked "Would it be ok if the paramedics cut up your friend to save 5 people who need organ transplants?", you would say "No". Trying to explain the general principle behind saying "No" is much harder: people with different ethical frameworks will differ on the explanation). 

Approval-directed training is bespoke in that feedback depends both on the algorithm A and on the previous history of A's actions. The more general property of approval-directed training is that the human feedback depends on a large amount of context or state. For instance, the human might be taking into account not just the algorithm and its history but a complex, high-dimensional representation of the local environment. Suppose the algorithm is controlling a robotic arm. Then the human may be aware of the position of the arm, its material composition, the conditions in the room around the arm, and so on. This is quite different from standard ML datasets for object classification or speech recognition. For these datasets, humans can label the datapoints without knowing the context in which the picture was taken or the context in which the speech was uttered. 

What are some Machine Learning problems relating to bespoke feedback? 

1. Relate bespoke training to training with side-information. Bespoke feedback will generally be expensive, high-quality training data. A learning algorithm might make use of cheaper signals in addition to occasional bespoke feedback. These cheaper signals might be training data that is not bespoke or else bespoke feedback that was intended for a different algorithm. Relatedly, consider the case where the learning algorithm A has a model of how the bespoke feedback works and so recognizes that the feedback it gets is a function of its previous actions. (As it scales up, a learning algorithm might need to understand this). 

2. Try out some applications of this approach. One question is how big/heterogeneous can the context be in terms of what the learning algorithm can represent as the context.

------------
------------

<a id="motivation"></a>

## 2. Motivation from value learning

It's easy to connect learning from side-information to existing research in ML. Here are some connections:

1. Semi-supervised learning or active learning in situations where there are some labels but additional labels are expensive. 

2. Dealing with noise in the training data; typically the noise is in the target variable for prediction ("dealing with label noise").  

3. Vapnik's work on "learning from priveleged information". Here the training examples for a supervised task come with "priveleged information" (a bit like side-information) that is not available at test time. In some cases, this might be information that is related to the algorithm used for predicting (e.g. information that is expected to help SVM). In the guiding examples, the priveleged information is high-dimensional and helps learn a mapping from X to Y (rather than being a noisy or transformed version of Y itself). 

While it's a viable project for basic research in ML, I'm not sure people would see it as a high priority. There has been lots of work on semi-supervised and active learning but they don't appear to be used much compared to standard supervised learning.  Maybe it's partly that they don't combine that well with the best supervised learning algorithms (or maybe they are not worth the hassle in general). In any case, ML people might be skeptical of the practical relevance of learning from side-information (as a pure ML problem) due to its similarity to active and semi-supervised learning. (For recent work on semi-supervised learning, see the deep generative approach of [Kingma et al](http://papers.nips.cc/paper/5352-semi-supervised-learning-with-deep-generative-models) and [Schoelkopf](https://scholar.google.com/scholar?cluster=17859746430977271826&hl=en&as_sdt=0,5) on causal learning and semi-supervised learning.

Some approaches to value-learning look like they *need* some form of learning from side-information. So if such learning is less effective in some fundamental way than supervised learning, this would be good to know about. Unfortunately, my guess is that it's very hard to show that active learning or semi-supervised learning can't work much better than they currently do with continued research.

For now, I will sketch why value-learning would benefit especially from learning from side-information. I find it plausible that either existing semi-supervised or active learning techniques can enable good enough learning from side-information or that incremental progress in existing techniques will suffice. Here are the motivations from value-learning:

1. Building on the discussion of approval-directed approaches above, feedback that is bespoke and has a high-dimensional and heterogeneous context (i.e. the context can't easily be mapped onto a lower-dimensional manifold) will be expensive. For example, each judgment might take minutes for an AI researcher who is an expert in the relevant algorithms. So the learning algorithm will need to use cheaper proxy signals and do active learning to supplement high-quality bespoke feedback. 

2. Suppose the goal is to teach a learning algorithm human preferences in some domain. Preferences are typically holistic (i.e. preferences over wholes are not a simple function of preferences over parts). Moreover, preferences are subjected to reasoned deliberation: our "best guess" can vary non-monotonically with further deliberation. This raises difficulties for the idea of creating a *single* training set that can be used for supervised learning. The preference judgments/labels in such a training set would be subject to non-monotonic changes under further deliberation and so would not be definitive. Still if are working with a single supervised training set, one step in the right direction is to distinguish the quality of labels based on the amount of deliberation/reflection and experience that the human has. In the approval-directed approach with bespoke training, the aim is not for the feedback to be a definitive expression of the human's preferences. Instead, the learning algorithm will ultimately understand the human's feedback as being about what counts as a good action in a very particular context. As the context shifts, the algorithm will be aware that the human's previous feedback is not "the last word" and that they might judge the present context differently. 

3. There is lots of lower quality data available that pertains to human preferences. For example, any kind of data that records human actions can be used to infer preferences via IRL. There is also lots of data that is a direct but noisy expression of preferences: e.g. Facebook Likes, Twitter retweets, Reddit upvotes, the world values survey, opinion polls, questionnaires. Finally, if the goal is to predict the preferences of a particular person, we can get lots of relevant information if we know about the preferences of other people.

As we noted above, getting high quality data about preferences is often expensive. So finding ways to learn about high-quality judgements via the kind of data that is already abundant and cheap would be desirable. In doing so, it is important to deal with the limitations of the cheap data. A reliable system for recommending actions should not recommend smoking to someone who smokes but wishes to quit. (In contrast, if object recognition for Google Image Search is bad at discriminating between two subspecies of hamster, this is not a big deal for use of the service in general). Distinguishing between activities like smoking and activities that are almost always benign (e.g. financial planning) may not be possible from the cheap source of data alone.

There are similar issues for recommending actions to a collective rather than an individual. Tim Scanlon has an example where 100,000 people are watching a game of soccer on TV. The TV broadcast of the game might cut out for a few seconds in the middle of the action. The question is whether it's ok to torture one person in order prevent the broadcast cutting out. Most philosophers (as well as conventional morality) would say that it's bad if the man is tortured. To extend the example, we can replace the torture with something that most people don't mind but which is like torture for a small minority of people (e.g. imagine an enclosed space for a claustrophobic). So you could know the preferences of the 100,000 viewers about watching the soccer, but because you don't know about the unusual fear of heights of a single person, you recommend a morally bad outcome. 

<a id="concrete"></a>

## 3. Concrete Problems in Learning Efficiently from Human Feedback

### Introduction
In order to make progress on these problems, there are two broad approaches. One is to create a simple, abstract, ML-style problem based on "learning from side-information". This is what Christiano does in his eponymous blog post. One benefit of this is that the abstracted problem is simple to describe and often easier to reason about analytically. Moreover, one can investigate it by using either synthetic data or a standard ML problem like MNIST. 

The second approach is to work on problems that are closer to value-learning. This probably entails more work setting up the problem and more work in a paper explaining the problem. On the other hand, results may help more directly with value-learning work and the application might make it clearer to our audience why we are interested in learning from side-information.

### Abstract prediction problems with side-information
The goal here is to investigate the abstract prediction problem described in Christiano's post. This setting is like semi-supervised learning but with the addition of z-variables. Paul sketches some algorithms that will improve performance by reducing variance. I believe that related general arguments have been made in the case of semi-supervised learning (Fox-Roberts' recent JMLR paper). One could adapt some of these arguments, with the goal of understanding what kind of improvements can be expected by making use of side-information rather than just the Y labels. For the passive learning case, a baseline is running a semi-supervised algorithm (assuming it improves on a purely supervised approach). For the active learning case, a baseline is running an active learning algorithm (assuming it improves on a purely supervised approach). 

The goal here is to consider whether good performance can be achieved with much less labelled data if side-information is used intelligently. This is not a well-specified problem without filling in more of the details. The crucial questions are:

(1). How cheap or prevalent is the side-information (e.g. what is the probability pi of observing the variable zi)? If it's too expensive or rare, then straight supervised or unsupervised learning will probably be best.

(2). How useful is the side-information for an agent who is fully informed of the joint distribution of X, Z and Y, compared to an agent fully informed of the joint distribution of X and Y on their own? If it's not very useful, then using standard supervised learning will probably be best. If the side-information is usually available and it's a noisy copy of Y, then we can just treat the side-information as a label and use supervised learning. Note that Lawrence and Schoelkopf (2001) find that unless label noise is very high, standard supervised learning will still perform well. 

Different settings of (1) and (2) could lead to a wide range of problems. So it might be important to find settings that are close to the applications in value-learning that provide motivation. The biggest source of variation is the way in which the side-information is helpful. Here are two examples to illustrate the variation:

1. Suppose MI( (X,Z), Y) > MI(X,Y), where "MI" is the mutual information. This condition is not sufficient for the side-information being useful because the extra information provided by Z needs to be in a form that makes it easy for a learning algorithm to extract. The simplest case here is where one of the z-variables is like an extra feature when added to X. In this case, supposing the z-variable is cheap, we might always pay to have access to it at test-time (since we'd do much worse without access to Z). 

2. Suppose the mutual-information condition above fails (since Z provides no extra information about Y than X alone does) but that MI(Z,Y) is substantial. For example, we might imagine that Y is a deterministic function of X with a small amount of independent random noise and that Z is a copy of Y with its own independent random noise. Here, the benefit of Z is simply in helping us to learn the deterministic function from X to Y. Once we have learned that function, we never need pay for Z.

One approach to fixing on a setting of (1) and (2) is to look at the value-learning related problems and try to abstract from them. In one such value-learning problem, the z-variable are preference judgments after different amounts of reflection or else they are judgements from people with different levels of expertise/experience. One abstraction of this problem to suppose that the z-variables are copies of Y with noise that is independent and has higher variance. This will probably make for a fairly easy ML problem but fails to capture the difficulty of the original problem. The variation in the preference judgments is mostly not from random noise (or anything that has similar statistical properties) but is more systematic and structured. Here are some examples that might be a better analog:


>**Prediction problems with side-information from humans**
>We call this "Prediction from Turkers vs. experts". The data is like MNIST but as well as numbers and Latin letters there are mathematical symbols (Greek letters and perhaps some symbols like '>' and '#'). Even if people are given a drop-down menu, some Greek letters (alpha, beta, omega, rho) are not easy to distinguish from their Latin cousins. And we'd expect more mistakes from Turkers than from mathematicians. Maybe Brendan Lake's omniglot data has some classifications from native speakers of the language as well as people who just learnt the characters. 

>There are many similar examples. You could take any classification task where Turkers can do the labelling and then produce z-variables of different quality by varying the time-limit for labelling and the amount of practice/experience the Turkers get before generating actual labels. (You could also try degrading the input data to simulate the effect of human's having less expertise). 

>Another simple schema: for a task that is linguistically demanding, create a z-variable from non-US Turkers, where you expect that difference in language ability will lead to some systematic differences in label quality. One example: 'evaluate the readability and grammaticality of these English sentences on a scale of 0-10'.

>On a practical note, it'd be a lot of work to produce a whole new dataset. If we wanted to collect new data at all, the best way would be to use some existing dataset and get some cheaper labels for it. 

These prediction problems are analogous to value-learning from humans with different amounts of deliberation. Another value-learning setting has training/feedback from both an actual human and from an algorithm B that mimics the human on some subset of contexts. The idea is to use this data to train the algorithm A, which can have overall superior performance to algorithm B. By abstracting away the value-learning details, we can construct ML prediction problems like those above:

>**Predictors trained on predictors:**
> Take some of the labelled data from a prediction problem. Construct an algorithm B that achieves good performance (close to the human) on an easily identifiable subset of instances. The problem is to train an algorithm A from the original labelled data plus the side-information provided by B (either actively or passively). 

People must have tried approaches similar to this. In semi-supervised learning, there is the technique of "self-training" that uses the same learning algorithm for both B and A and uses B to label all the unlabelled instances. This is different from our more general case. Our algorithm A treats B's labels as side-information (rather than as real labels). And we don't assume that A and B have anything in common. (Still, if A and B were the same learning algorithm we'd be able to define a recursive bootstrapping procedure.)

There are various ways to construct the algorithm B. You could train an algorithm B that outputs uncertainty as well as predictions. Assuming B is empirically well-calibrated, you could use B's labels only when it has confidence above a threshold. A different approach is to train B on an easier version of the whole problem. For example, for object recognition, you could train B to distinguish between images that contain writing (e.g. signs, banners, T-shirts with writing) and images that don't. Algorithm A would then have these labels as side-information. If B is similar to A and is given a similar to task to A, then we don't expect B's labels to help A much. So we want to consider either B being different from A (e.g. A is good at computing posteriors given small labeled datasets and B is good learning efficiently from large labelled datasets.) or B's task being different. 




### Dialogs and social networks
Having discussed abstract versions of learning from side-information, I will now consider problems that are closer to the value-learning setting. 

#### Feedback on tree-structured dialogs
(Andreas can fill out the details here). In this setup, the goal is to evaluate the usefulness of a contribution to a tree-structured dialog aimed at answering the question at the root node. Evaluations of contributions are either given during the dialog (without the benefit of knowing the dialog's final state) or after the dialog has been closed (when the asker might have reflected further on the question and the value of the dialog). We'd like to use the quick, cheap evaluations (from both the asker and answerers) that happen during the dialog to predict the post-dialog judgments of the asker. The benefit of this is not just faster and cheaper prediction but also a chance to shape the course of the dialog in a path-dependent way (e.g. by removing contributions that -- with the benefit of hindsight -- would have taken things down a bad path).

#### Facebook "Likes"
There is a related problem for FB-like social networks. Suppose people are shown content based mainly on what friends share. You allow them to provide public ratings (Likes, etc.) and get information about what they click on. You can also collect more thorough/careful evaluations of content. These could be done by trained experts or by random members of the social network. The idea is to show people some content and ask them a few questions about how rewarding and enjoyable they found the content. The content could be something they already saw, which allows them to judge with the benefit of hindsight -- e.g. was the event someone linked to actually enjoyable to attend? The content could also be something new. The questions would be optimized to probe (a) whether people find the content rewarding in a long-term sense, (b) whether they'd look at the content simply to please a friend (so they should only be shown the content if it comes from a friend they care about). Maybe it's worth asking whether people prefer the content to other posts and to other activities (talking to the friend, watching a movie, doing work). You could ask for verbal explanations to encourage deliberation. 

The idea is then to treat Likes and other behavior (shares, comments, click-throughs) as side information for predicting the thorough evaluation of the content. As with the case above, the Likes will not just be a copy of the thorough evaluation with independent noise. Some kinds of content will look good superficially and be more likely to elicit over-hasty Likes that wouldn't withstand deliberation. People will also choose whether to Like for various social reasons. These include:

1. communicating to friends that they've seen some content (without bothering to comment)

2. rewarding friends by pretending to like content (esp. if the content has very few likes from others)

3. communicating that you like this kind of content (even if you've already seen this particular piece of content elsewhere)

4. wanting to avoid communicating that you like a certain kind of content (because friends would disapprove) even if you do.

So the true generative model for how Likes and thorough evaluations are related will be complex. However, one can imagine learning a simplified version of it. A simple approach would be to assume that the noise in Likes (and other side-information) is a function of the content itself and of the closeness of the friend who posted it. The intuition is that Likes become noisier if the content is more personal (e.g. status update rather than linked NYT article) and if the friend is closer. For background on Facebook newsfeed, see Slate article "Who Controls Your Newsfeed?" by Will Oremus


### Problems involving RL

The abstract prediction problem based on "learning form side-information" is limited in that it defines a task that only consists of prediction. To construct a scaleable, safe, value-learning AI system, it is desirable that the system take *actions* rather than just making predictions. It might be that a large part of the problem of taking good actions can be reduced to making good predictions. So the former problem may be very useful for helping with the latter. (Some versions of Approval-directed agents might be like this. The agent doesn't do any explicit planning and just takes the action that it predicts will get the most approval.) However, it seems worth working on both prediction and action-selection problems as parallel research programs.

As with prediction problems, we can consider RL problems that abstract away some of the details of the value-learning (or approval-directed agents) problem. The benefit is again to get a simpler problem that can be tackled with pre-existing tools. Christiano discusses this kind of approach in the post "Semi-supervised Reinforcement Learning". I will discuss these abstrated problems, as well as problems that get closer to the value-learning problem. 

#### Abstracted RL problem

Many practical RL problems involve a human trying to control an AI's behavior by setting up the RL problem in the right way. The methods of human control are fixing the reward function and fixing the RL algorithm used by the AI. (The human might also select environments that are apt for learning).

The simplest approach is to take some standard RL problems and reduce the quantity or quality of the reward signal. This is what Christiano suggests in his post. For example, you can take problems like Gridworld, Atari, board games, etc. and change the pre-specified reward functions to make it a "learning from side-information" RL problem. 

*Learning from side-information: from prediction to RL*
<br>
It's worth relating this RL problem to the prediction version of learning from side-information. In the prediction problem, you have to predict Y from X, where you mostly haven't observed X and Y together in the past but you may have observed side-information Zi that's helpful for inferring Y from X. The probability or cost of observing Z or Y given input X may vary depending on Y or on X. (Also: depending on the problem setup, it may be that Zi and Y can be queried for any possible value of X or else that the active learner has to make do with the values of X in the training set). 

The natural counterpart in the setting of RL is to suppose that the agent does not always observe the true reward R(s) in a state s but may observe side-information zi that pertains to R. The case where the reward function R(s) is stochastic is already familiar in RL (e.g. in Bandit problems). In these cases, the agent gets iid samples from R(s) when visiting R, and so simply visiting state s repeatedly will normally suffice. Whereas in the version with side-information, it could be that R(s) is never observed in some states s and so the agent must rely on side-information and on generalization across states.

[There are more notes on RL problems in another document.]




<!--
*Key differences between the prediction and RL problem*
It's also worth highlighting how the RL problem with side-information differs from the prediction problem with side-information. In the prediction problem, the agent may have a fixed set of X values in the training set or they may be able to request iid samples from X or specific values from X to be labelled. In the RL problem, on the other hand, the agent generates their own non-iid "training data" by moving between states. The problem of making plans that lead to (a) lots of information about R, and (b) high reward may be much trickier than active learning in the prediction problem. This is a standard difference between RL and prediction problems. We could alter the RL problem further by requiring the agent to move to a special "oracle" state in order to pay for data about R (e.g. where the state is a particular grid location in gridworld). It seems likely that variation among states in terms of observing R(s) and the side-information will make this trickier but it's not clear. (Model-free strategies for exploration such as epsilon-greedy or optimistic initialization may work poorly if R(s) is not observed in some large subset of states and this subset is identifiable).

*Concrete RL problems with side-information*
The simplest kind of problem is "semi-supervised RL", where the agent only observes the reward function in certain states. In this case, the agent can only infer anything about the reward of other states from its prior on reward functions. As with semi-supervised learning, this seems very hard in general.

An easier problem is where the agent receives side-information Z*(s) in state s which is a subset of the components of Z(s) = (z1(s), z2(s), ... , zN(s)=R(s)). In the simplest case, the agent knows that the z-variables potentially provide information about R but that they are systematically or randomly noisy. To implement this, we could take a Gridworld problem and construct a model of the Z-variables given s and R(s). The RL agent might be model-free, in which case it would either treat the Z's as noisy rewards or they'd be incorporated into the state representation. Or else it might have some accurate model of how the Z-variables relate to S and Y without knowing all the relevant parameters. My very rough guess is that an approach using dQN, where the Z-values are treated as features used to help predict the Q-value of a state would work better than a model-based approach (unless the model-based approach had enough of the model built in already). Still, as noted above, it seems that Q-learning would struggle if most states don't give much info about their reward and substantive planning is needed to reach the states that do (rather than exploration via epsilon-greedy). 

To bring the problem closer to value-learning, we can consider ways of constructing the side-information that are more fitting -- as in the example of classifying handwritten math symbols for Turkers vs. mathematicians. We could take a simple driving simulation game of the kind that's been used for IRL experiments. The goal here is to drive in ways that humans approve of. We could have humans evaluators provide reward values by directly inspecting different states. (We can model reward on legal/moral weight of different states. So collisions, speeding and driving on the wrong part of the road are all bad. Driving much slower than nearby cars is also bad). 




As Paul suggests, we can take any standard RL problem and limit access to the reward function. For passive learning, we could noisify most of the reward (either randomly or as a function of state). To match the learning from side-information problem, the agent might or might not receive reward y and would get some amount of side-information z (which will be like a POMDP observation -- it doesn't contribute to the transition function but just gives info about reward). For active learning, we give the agent a special action (which might not take any time --- but would have a certain cost) and have them decide whether to buy the side-information on a given occasion. 

This is nice because (a) we can use existing code, (b) easy to argue that this is relevant.

#### IRL Problem
One version: you get entire trajectories of different quality. We can think of these as actions after more experience/reflection. (Formally, could also be more or less biased/bounded humans -- varying across people). You could also have meeting halfway (bespoke data) which is meant to be easier for the AI but isn't exactly the reward function. 

We can imagine different generative models. One would be that the utility function is gets noised up, resulting in some significant differences in behavior. Another that the noise consists of ramping up biases or bounds (more discounting, shorter time horizon and so on). We can still no difference despite these different settings (though using multiple trajectories will be quite taxing).

Another version is where the biases/bounds vary across individual actions. So you just throw in state-action pairs and give them a label based on the conditions. (This could be for fleeting biases/bounds --- e.g. icard and my probabilistic hyperbolic discounting. Or a case where the time horizon of the agent varies over time).

In terms of the generative model, we can imagine fixing the utility function and then allowing the bias and bound parameters to vary in the different settings. So we can learn how informative each of the settings are. (Imagine we don't have many unbiased settings and we have to learn the preferences mainly from the biased settings. One case is where we know the bias. This could still make things hard because the bias/bound might make some things hard to identify. On the other hand, if we can learn the bias, then some things might become *easier* to identify (e.g. the hyperbolic discounting case).

This seems like a quite natural extension of what we're doing. There's lots of data of behavior where there might be biases. If you get people to consider their actions carefully (maybe after de-biasing) you should reduce biases. We'll tend to have more of the possibly biased data.

This still has the basic problem of the difficulty of representing and capturing utilities accuratley. So what about learning instrumental values or learning imitation? If your aim is mainly to learn a good model of behavior, and you are not that concerned with transfer learning (and representing preferences in a way that facilitates transfer), then the above still seems relevant. The behavior you want to imitate (or the instrumental values you woant to learn) are those of the agent with lots of experience/reflection time. 





-->
