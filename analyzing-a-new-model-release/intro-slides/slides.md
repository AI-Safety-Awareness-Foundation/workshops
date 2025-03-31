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
    * Useful proxy for domain-independent
+ Increasingly used in industry
    * Benchmarks are annotated with amount of time human experts took to
      complete problem
+ Month-long point seems like threshold for "crazy AI"
    * AI that is capable of doing useful intellectual tasks better than human
      experts
    * Maybe not that far off! (Uncertainty remains!)

# Benchmarking focusing on software development

+ Top AI companies are focusing on software development benchmarks
    * Racing to self-recursive improvement (AI that can autonomously make itself
      better)
    * Likely inflection point
        - Might go doubly exponential improvement (can recursively improve both
          AI model itself and the process of improving itself)
+ Potentially one of the most important capabilities for an AI system
    * Also very dangerous!

# Company incentives

+ Why is the company releasing this model now?
    * E.g. 4o image generation was announced in March, 2024
    * Why did they finally release in March, 2025?

# Potential for less transparency in future

+ It is likely that as models progress, less public information will be known
  about them
    * Models that accelerate AI R&D can be more valuable if kept within a
      company vs money made from releasing it
+ This next year is crucial for journalists!

# Deciphering model numbers

+ "Llama-3 is great"
    * But what exactly is "Llama-3?"
    * When we go to meta

# Where should I look for info

+ System card/model card
    * This is provided by the company
    * *Not* the one provided by HuggingFace or other websites (although those
      can be worth looking at too, they are more tailored to software developers
      and often omit important details)
    * This is often unfortunately not very standardized and is kind of
      idiosyncratic and random where a company will put it
        - E.g. Google DeepMind often does not provide this at all
        - Generally OpenAI and Anthropic will accompany each release with one
        - But the format is not standardized at all
        - If you can't find it, can directly ask a lab for a model card/system
          card
+ 3rd party evaluation organizations
    * UK AISI/US AISI
    * METR
    * Apollo Research
+ Twitter
    * Usual wheat vs chaff issues
    * Nonetheless surprising overrepresentation of AI field on Twitter
