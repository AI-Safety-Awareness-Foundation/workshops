# How to think about analyzing a new model

A new model has been released. Great now what?

+ What questions should I be asking?
+ What sources should I go to?

# What are crucial questions I think about?

+ Is this in line with expectations?
    * If so, what do my expectations predict about AI over the next 36 months?
    * If not, how should I adjust my expectations?
+ Is company information consistent with other 3rd party evaluators?
+ What does this tell me about company incentives?
+ You might have others!
    * But would recommend at least thinking about these

# How do I figure out what my expectations should be?

+ Go over popular benchmarking approaches today and understand them

# Time-based AI benchmarking

+ Time as a measure of difficulty
    * Can this model do a task that takes a human expert 1 second? 1 minute? 1
      hour?
    * Time helps compare tasks, even if they're very different (e.g. translating a sentence v. solving a math equation)
+ Increasingly used in industry to describe performance 
    * Benchmarks are annotated with amount of time human experts took to
      complete problem
+ Month-long point seems like threshold for "crazy AI"
    * AI that is capable of doing useful intellectual tasks better than human
      experts
    * [Maybe not that far off](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/)! (Uncertainty remains!)

# Software-development ability based benchmarking 

+ Top AI companies are focusing on AI coding skills 
    * Key inflection point: AI that can improve itself
    * TK: what are the main hurdles they've yet to overcome? 
+ Potentially one of the most important capabilities for an AI system
    * Also very dangerous!

# Company incentives

+ Why is the company releasing this model now?
    * E.g. 4o image generation was announced in March, 2024
    * Why did they finally release in March, 2025?

# Potential for less transparency in future

+ As models progress, labs may release less information to the public **BECAUSE [TK]**
+ This next year is crucial for journalists **BECAUSE [TK]**

# Deciphering model numbers

+ "Llama-3 is great"
    * But what exactly is "Llama-3?" **[COULD ADD AN IMAGE FROM HF TO EMPHASIZE # OF VARIANTS]**
    * When we go to Meta **(unsure of what's being said here)**

# Where should I look for info

+ System card/model card
    * This is provided by the company
    * *Not* the one provided by HuggingFace or other websites (although those
      can be worth looking at too, they are more tailored to software developers
      and often omit important details)
    * There's no standard format or location:
        - E.g. Google DeepMind often does not provide this at all
        - Generally OpenAI and Anthropic will accompany each release with one
        - But the format is not standardized at all
        - If you can't find it, can directly ask a lab for a model card/system
          card (and if they don't have one you can ask "why not?")
+ 3rd party evaluation organizations
    * UK AISI/US AISI
    * METR
    * Apollo Research
+ Twitter
    * Usual wheat vs chaff issues (?)
    * Significant representation AI field on Twitter (good point, but how to validate legitimacy?) 
