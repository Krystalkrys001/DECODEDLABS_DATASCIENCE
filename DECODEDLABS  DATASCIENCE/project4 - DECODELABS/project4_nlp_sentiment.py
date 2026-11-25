# =========================================
# PROJECT 4 (OPTIONAL): NLP & Sentiment Analysis
# DecodeLabs Data Science Industrial Training, 2026 Batch
# =========================================

import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# =========================================
# STEP 1: Load and inspect
# =========================================
df = pd.read_csv("amazon_reviews.csv")
print(df.shape)                          # (20000, 2)
print(df.isnull().sum())                 # 0, 0
print(df["Positive"].value_counts())
print(df["Positive"].value_counts(normalize=True) * 100)
# 76.17% positive, 23.83% negative, a real, milder imbalance than Project 2's fraud data

# =========================================
# STEP 2: Text pre-processing
# =========================================
# Keep negation words OUT of the stop-word removal list.
# Default stop-word lists remove "not", which flips sentiment entirely
# ("I am not happy" -> "I am happy") if left unfixed.
stop_words = set(stopwords.words('english'))
negations = {"no", "not", "nor", "never", "none", "n't", "cannot", "isn't", "wasn't",
             "aren't", "weren't", "don't", "doesn't", "didn't", "won't", "wouldn't",
             "shouldn't", "couldn't", "can't"}
stop_words = stop_words - negations

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith('J'): return 'a'
    elif tag.startswith('V'): return 'v'
    elif tag.startswith('R'): return 'r'
    else: return 'n'

def preprocess(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)            # strip HTML tags like <br>
    text = re.sub(r'[^a-z\s\']', ' ', text)        # strip punctuation, keep apostrophes
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tagged = pos_tag(tokens)
    lemmatized = [lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged]
    return " ".join(lemmatized)

df["clean_review"] = df["reviewText"].apply(preprocess)

# EDGE CASE FOUND: one review was written letter-spaced ("n o t. w o r t h."),
# so every token was length 1 and got filtered out entirely, leaving an empty
# string. On save/reload, pandas silently turned that into NaN. Fixed here:
df["clean_review"] = df["clean_review"].fillna("")

# =========================================
# STEP 3: Train/test split (stratified, before vectorizing)
# =========================================
X = df["clean_review"]
y = df["Positive"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================
# STEP 4: TF-IDF vectorization
# =========================================
vectorizer = TfidfVectorizer(max_features=5000, min_df=2, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)    # fit only on train
X_test_tfidf = vectorizer.transform(X_test)           # test only ever transformed

# =========================================
# STEP 5: Baseline model (MultinomialNB)
# =========================================
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
y_pred = nb.predict(X_test_tfidf)
y_proba = nb.predict_proba(X_test_tfidf)[:,1]
print("\n=== Baseline: MultinomialNB ===")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
# RESULT: 97.44% positive recall, but only 60.97% negative recall.
# The class imbalance skewed the model toward the majority class.

# =========================================
# STEP 6: Final model (ComplementNB)
# =========================================
# ComplementNB is purpose-built by its creators for imbalanced text
# classification. Tested against SMOTE + MultinomialNB as an alternative;
# both produced near-identical improvement, so ComplementNB was chosen
# as the simpler fix, no extra resampling library or step required.
cnb = ComplementNB()
cnb.fit(X_train_tfidf, y_train)
y_pred_cnb = cnb.predict(X_test_tfidf)
y_proba_cnb = cnb.predict_proba(X_test_tfidf)[:,1]
print("\n=== Final: ComplementNB ===")
print(confusion_matrix(y_test, y_pred_cnb))
print(classification_report(y_test, y_pred_cnb, digits=4))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_cnb))
# RESULT: negative recall rose from 60.97% to 87.93%, ROC-AUC held steady
# at ~0.946, confirming the model's underlying separating power didn't
# change, only where it draws its decision line.

df.to_csv("amazon_reviews_clean.csv", index=False)
print("\nSaved cleaned dataset with predictions-ready features.")
