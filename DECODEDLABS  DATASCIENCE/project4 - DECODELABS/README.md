# Project 4 (Optional): NLP & Sentiment Analysis
**DecodeLabs Data Science Industrial Training, 2026 Batch**

## Overview

This project trains a machine to read product reviews and decide, purely through math, whether each one is positive or negative. Marked optional in the training track, it was completed anyway to push past dashboards and structured data into unstructured text, a genuinely different discipline.

**Dataset:** 20,000 real Amazon product reviews with binary Positive/Negative labels, sourced from PyCaret's public NLP datasets repository (the same dataset used in PyCaret's own official tutorials). Class split: 76.17% positive, 23.83% negative, a real, milder imbalance than the 0.17% fraud rate in Project 2.

## 1. Text Pre-Processing

Raw review text cannot be fed to a model directly, it has to be cleaned and standardized first. The pipeline: lowercase, strip HTML remnants and punctuation, tokenize into individual words, remove filler words, and lemmatize each word to its true dictionary root using part-of-speech tagging (so "went" correctly reduces to "go," not left as-is the way cheaper stemming would mangle it).

**The Stop-Word Trap, caught and fixed:** default stop-word lists remove "not" along with every other common filler word, since it's technically frequent. Left unfixed, "I am not happy" silently becomes "I am happy" after cleaning, flipping the sentiment entirely. Fixed by explicitly removing negation words (not, never, don't, isn't, and their contractions) from the stop-word list before applying it. Verified directly: `preprocess("I am not happy with this product.")` correctly returns `"not happy product"`.

**A real edge case found and fixed:** one review in the dataset was written letter-spaced for emphasis ("n o t. w o r t h."). Every resulting token was a single character, and the pipeline's length filter stripped all of them, leaving an empty string. On save and reload, pandas silently converted that empty string into `NaN`, which broke vectorization downstream. Fixed with `.fillna("")` before proceeding, the same tool used for the missing CouponCode values back in Project 1, applied here for a different root cause.

## 2. Vectorization (TF-IDF)

Cleaned text was converted into numerical weights using TF-IDF (Term Frequency, Inverse Document Frequency): common words across all reviews (like "product") are automatically down-weighted, while words that are both frequent in a specific review and rare overall (like "terrible") are weighted heavily. Configured with `max_features=5000` and `min_df=2` to cap vocabulary size and filter out one-off typos, and `ngram_range=(1,2)` to capture two-word phrases like "not good" as their own feature, not just "not" and "good" separately.

## 3. Modeling: Baseline vs. Final

| Metric | Baseline (MultinomialNB) | Final (ComplementNB) |
|---|---|---|
| Negative Recall | 60.97% | **87.93%** |
| Negative Precision | 88.16% | 68.74% |
| Positive Recall | 97.44% | 87.50% |
| ROC-AUC | 0.9462 | 0.9462 |

The baseline MultinomialNB model, trained on the imbalanced data as-is, caught 97% of positive reviews but only 61% of negative ones, the class imbalance skewed its decision boundary toward the majority class. Two legitimate fixes were tested side by side: ComplementNB (a Naive Bayes variant purpose-built for imbalanced text classification) and SMOTE resampling combined with standard MultinomialNB. Both produced near-identical improvement.

**ComplementNB was selected as the final model.** It achieved the same gain as SMOTE, negative recall rising to 87.93%, without introducing an additional resampling library or step. ROC-AUC stayed essentially unchanged (0.9462 either way), confirming the model's underlying ability to separate the two classes never actually changed, only where the decision boundary was drawn.

## Confusion Matrix (Final Model)

```
[[ 838  115]
 [ 381 2666]]
```

838 negative reviews correctly caught, 115 missed as false positives. 2,666 positive reviews correctly caught, 381 wrongly flagged as negative, an honest trade-off, not a flaw hidden from the write-up.

## Tech Stack

Python, Pandas, NLTK (tokenization, stop words, WordNet lemmatization, POS tagging), Scikit-learn (TfidfVectorizer, MultinomialNB, ComplementNB), imbalanced-learn (SMOTE, tested as an alternative)

## Files

- `project4_nlp_sentiment.py` — full pipeline: load, pre-process, vectorize, baseline model, final model
- `amazon_reviews.csv` — source dataset
- `amazon_reviews_clean.csv` — dataset with cleaned text column

## Key Takeaway

The negation trap and the letter-spaced-review bug were both the kind of failure that produces no error message, the code runs fine and returns a plausible-looking wrong answer. Catching them required testing the pipeline against real edge cases before trusting it on the full dataset, the same discipline carried through from Project 1's coupon code decision to Project 3's cluster count, applied here to text instead of numbers.
