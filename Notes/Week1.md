# Week1: Neural Language Model Basics

## Learning

1. Language Model

    Mathematically, language models are to solve $P(y_t | y_{1:t-1})$.

2. Statistical N-Gram Model (see [example codes](/Notebooks/week1.ipynb))

    Probabilities are computed by taking some corpus of text (n-gram) and counting how often different sequences of words exist.

    Statistical language models have two key problems:

    - Statistical language models have an exponential growth in model parameters as the n-gram size is increased.
    - Statistical language models do not handle data sparsity well.

3. Neural Language Models (see [nanoGPT](https://github.com/karpathy/nanoGPT) or [Annotated Transformer](/Notebooks/week2.ipynb) for Transformer implementation)

    Neural language models replace the giant lookup take of n-gram frequencies with a neural network trained to predict likelihoods. Neural language models solve both the problems describes in the previous section:

    - Neural language models can compute likelihoods of long sequences without the number of model parameters exploding.
    - By leveraging continuous representations and generalizing across word sequences, neural models can predict probabilities for sequences they’ve never explicitly seen in their training data.

4. Decoding Strategy

    - Greedy, $argmax_i \: P(y_t=i | y_{1:t-1})$
    - Random Sampling with temperature $T$
        $$P(y_t=i) = \frac{exp(z_i/T)}{\sum_j exp(z_j/T)}$$
    - Top-k Sampling
    - Nucleus Sampling (top-p sampling)
    - Beam Search (lacking surprise and creativy)

    Other parameters for decoding control:

    - Frequency Penalty
    - Presence Penalty
    - Stopping Criteria (token counts + special tokens)

    Also, [constraint decoding](https://github.com/Saibo-creator/Awesome-LLM-Constrained-Decoding?tab=readme-ov-file#related-awesome-lists) of LLM is important to get structured outputs.

5. Transformer Architecture

    [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

    [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

    [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)

## Interesting Notes from Piazza

- Non-determinism with LLMs / multi-GPUs

    The fundamental problem is that floating point arithmetic is not associative -- if you add, multiply, divide things in different orders the results are different. ([stackoverflow](https://stackoverflow.com/questions/10371857/is-floating-point-addition-and-multiplication-associative))

    This is combined together with the fact that GPUs are running many threads in parallel -- for instance when doing a matrix multiply, each sum of the rows/columns of the matrix may be done in a different thread. Depending on many factors (the temperature, power usage, etc.) these threads might finish at very slightly different times, so the results might be added together in different orders.

    This is exacerbated further when you're doing an argument like `argmax()` in generation. Even if there are tiny differences between the top two scoring tokens, at some point this might affect the maximum probability token, and once that token changes then the rest of the following tokens will be heavily influenced by having a different token in the context.

- Why Language Model generated text has different statistical characteristics than human text?

    There is a [paper](https://arxiv.org/pdf/1904.09751) that introduces Nucleus Sampling, analyzes this problem as __text degeneration__.

    Let’s just consider a pre-trained language model for now: that that has been trained for next-token prediction on internet data.

    If I pass in the prompt: “I set my cat down on the” to this LM, I’ll get a distribution that looks something like the one attached to the bottom of this post.

    Now, how do you decide what token the LLM should generate? Humans have communicative intent and pick their words based on what they are trying to say.

    The generation algorithm we're using with the LLM needs to make a decision based just on these probabilities. Choosing a high-probability word is low-risk. But picking the low-risk option all the time will yield text that is systematically less diverse than real-human text. In contrast, we could sometimes pick from the tail, but as you can see below, the long tail both has both good (e.g. "bunk") and bad (e.g. "bathroom) word choices in it that are nearly the same probability.

    As an aside: modern language models are typically pre-trained on internet text and the finetuned using a reward model that encourages them to produce outputs that would the reward model classifies as high-reward. Each company training these models defines "high-reward" differently, but usually, they optimize for some mixture of safety, helpfulness, lack of bias, etc. Unfortunately, this addition "alignment" step tends to result in models that have even peakier distributions than the pre-trained-on-internet-data language models. This makes it even harder to generate text that looks distributionally similar to what a human would write.
