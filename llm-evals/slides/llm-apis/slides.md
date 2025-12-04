# LLM API

+ For evals we usually focus on a standardized LLM API
+ API that you will normally see exposed in HTTP endpoint
+ One-level above raw model input output
    * Raw model input: tokens
    * Raw model output: one probability distribution per input token
+ API
    * API input: conversation
    * API output: conversation continuation
+ API design has remained fairly stable
    * Core of APIs hasn't really changed in 3 years

# Mental model

+ Man in a room
    * Gets a piece of paper
    * Piece of paper has words annotated in all sorts of different ways
+ Man reads it and is asked to complete the next set

# Is this just autocomplete?

+ Depends on how sophisticated your notion of autocomplete is!
+ But a naive understanding of autocomplete can cause problems
    * These models are *not* the "median of their input data"
        - Generally models tend to far outperform the median of their input
          data! (E.g. why did even GPT-3 in 2020 have basically perfect spelling?)
    * These models are *not* blank slates that mimic whatever input you give
      them
        - Models often have something emulating a "base personality" which is
          a major target of evals to uncover!
    * This misses the post-training step

# No real distinction between "agentic LLMs" and "base LLMs"

+ They're all just data in and data out!
+ Agentic LLMs are just about piping LLM outputs to tool inputs and tool outputs
  to LLM inputs

# Levels of spec

+ Company spec
+ System prompt
+ User prompt
