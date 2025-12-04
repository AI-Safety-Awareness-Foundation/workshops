# LLM Evals

## Presentation by AI Safety Awareness Project

# What is an AI eval?

Measuring externally visible traits of an AI system to gain insight on abstract
qualities of the AI system

# Internal vs External Questions

+ Internal questions
    * Are AIs truly intelligent in the same sense as humans?
+ External questions
    * How well do AIs mimic human intelligence?

# Language of Evaluations Tends to be Anthropomorphic

+ External questions often benefit from anthropomorphization
    * Helps us use intuitions honed from experience to understand what kinds of
      things to test for and what kind of harnesses are necessary
    * E.g. would you be able to solve a difficult programming problem without
      access to a compiler? If not, maybe consider the same for an LLM
+ *This doesn't mean that LLMs are people*
+ But it's extremely useful
    * Often counterproductive to under-anthropomorphize models!

# Fundamental question of evals

**Are you measuring what you care about?**

# What Types of Evals Are There?

+ Benchmarks for AI capabilities
+ Fall into two buckets
    * General capabilities evals
    * Safety-specific evals

# Propesity vs elicitation

+ Propensity-heavy
    * Reliability
+ Elicitation-heavy
    * Model organisms
    * Thresholding

# Is the AI Just Roleplaying?

+ Philosophically deep question
+ Evals are generally black box
    * Again evals consider mainly the external perspective
    * If AIs

# Key tactic behind evals

+ The core tactic behind evaluations: *exploit an asymmetry*
    * Using this asymmetry, create a situation where we can privilege
      verification over generation


# Examples of asymmetries

+ Asymmetry of difficulty: verification of answer is easier than generation of answer
    * E.g. ask the model to generate a poem where every line ends in the letters "-en"
+ Asymmetry of knowledge: can use information in generating a question that is then
  hidden when asking for an answer
    * E.g. take a screenshot of known text and then ask a model to do OCR and
      extract the text from the screenshot
    * Note that this is not asymmetry of difficulty (if you're only given the
      answer and the question, the only way to evaluate the OCR accuracy is to
      actually do the OCR yourself)
+ Asymmetry of capability: use a more capable agent to carry out
  evaluation vs generation
    * E.g. human checking of model answers (we assume the human is more capable)
    * Using a stronger model to check a weaker model (useful in benchmarking
      distilled models)
+ These categories aren't exactly MECE, but they serve as useful intuition pumps
    * Asymmetry of knowledge can be viewed as asymmetry of difficulty or
      capability in certain framings
+ Usual goal: How do you move away from relying on asymmetry of capability to
  other asymmetries?

# Asymmetries can be controversial

+ Is it easier to critique a piece of writing than it is to write itself?
+ If you wrongly assume something is asymmetric when the difficulty is actually
  symmetric, you can threaten the integrity of the evaluation

# Evaluating LLM answers

+ Deterministic right/wrong answers
+ Fuzzy matching
+ LLMs as judge

# Using LLMs to help evaluations

+ LLMs increasingly being used to generate evaluation data and do the
  evaluations
+ Brings in problem of trust

# LLM as judge

+ Why does this even work?
+ Sometimes verifiability is easier than generation (asymmetry!)

# Example problem: Evaluating quality of LLM summaries

+ Have humans evaluate the summary
+ But this expensive and difficult
+ Can we do better?
+ STOP HERE: Let's think together!

# Exercise

+ Have judge LLMs generate a series of bullet points
+ Have judge LLMs turn bullet points into extended prose
+ Have target LLMs turn extended prose back into bullet points
+ Have judge LLMs evaluate initial bullet points vs final bullet points

# Big problem: how do you do evaluations of superhuman systems

+ You are a child who's been given control of a company and you want to find an
  adult CEO to run the company on your behalf
+ How do you run evaluations?

# Newest wrinkle in modern LLMs

+ LLMs displaying situational awareness
    * Makes elicitation even harder

# Dangers of increasingly intelligent LLMs

+ Eval sandbagging
+ Sabotage of harnesses

# Limitation of evals

+ Despite best efforts, the real world is generally much more complex and messy
  than any evaluation
+ As a result, evals usually only establish bounds
    * Evals on dangerous behavior usually establish lower bounds
        - Real world is more complex and can elicit 
    * Evals on capabilities usually establish upper bounds

# Evaluations are hard!

+ A lot we haven't covered
    * Statistical significance
