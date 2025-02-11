# Week4: Evaluation of Language Models

## Learning

1. Word Overlap-based Metrics

    Following metrics could be used in evaluating sequence generation:

    - exact match / word error rate
        $$\mu(y, \tilde{y}) = \frac{\delta(y, \tilde{y})}{|y|}, \text{ where } \delta \text{ is word edit distance}$$

        No sematic mathcing makes it less corelated to human evaluation.
    - Perplexity
        $$
        \begin{aligned}
        PP(Y) &= P(y_1y_2...y_N)^{-\frac{1}{N}} \\
              &= exp(- \frac{1}{N} \sum_{i=1}^{N}\log P(y_i | y_{1:i-1}))
        \end{aligned}
        $$

        Though low perplexity in general represents better sequence generation, very low value could suggest non-human or LLM generated texts.
    - BLEU (especially for machine translation)
        $$\mu(y, \tilde{y}, k) = \prod_{i=1}^k(\frac{|S_i(y) \cap S_i(\tilde{y})|}{|S_i(\tilde{y})|})^{\frac{1}{k}} \text{, } S_n(y) \text{ is n-gram multiset}$$

        Typically, we use $k=4$ in above equation, and it measures the _precision_ of generation. BLUE has limitaion that ignores semantically similar words.
    - ROGUE
        $$\mu(y, \tilde{y}, k) = \frac{|S_i(y) \cap S_i(\tilde{y})|}{|S_i(y)|}$$

        Typically, we use $k=\{1, 2\}$ in above equation, and it measures the _recall_ of generation.

2. Modern Evaluation

    Metrics like BLEU and ROUGE are based on word overlap. However, these metrics may fail to capture the semantic similarity.

    - BERTScore

        BERTScore operates by calculating cosine similarities between the embeddings of tokens in the reference and generated texts.

    - COMET

        COMET (Cross-lingual Open-source Metric for Evaluating Translations) represents a shift towards metrics that aim to predict human preferences directly, rather than merely approximating them. COMET employs two primary approaches:

        - Regression Model: Trains a model to predict human ratings (e.g., Likert scale ratings) of text, based on historical human annotations.
        - Ranking Model: Trains a model to rank outputs based on human preference, such as ranking two translations based on which one humans prefer. The ranking approach has shown stronger performance in correlating with human judgments.
        - Advantages: COMET aligns more closely with human preferences by training on human annotations. However, it requires a significant amount of annotated data, limiting its applicability to well-annotated tasks.

    - Multitask Evaluation

3. MMLU: A Case Study

    MMLU (Massive Multitask Language Understanding) is an important benchmark used by nearly every modern language model to report performance results. It evaluates how well a model can handle a wide range of tasks across diverse domains.

4. Human Evaluation

## Interesting Notes from Piazza

- [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361) by OpenAI in 2020
