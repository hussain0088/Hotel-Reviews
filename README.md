# Hotel Review Sentiment Preprocessing (NLP)

## Overview

This project focuses on Natural Language Processing (NLP) for cleaning and preparing hotel review text data. The main objective is to perform a comprehensive set of essential text preprocessing techniques—such as **tokenization, stemming, lemmatization, and N-gram generation**—to ready the reviews for subsequent tasks like sentiment analysis or classification.

---

## 1. Dataset
The analysis is performed on a set of hotel reviews read from the file: `/content/hotel_reviews.csv`.

### Dataset Details
* **Total Entries:** 109 reviews.
* **Columns:**
    * `Review`: The raw text of the hotel review.
    * `Rating`: The numerical rating (1 to 5) associated with the review.
* **Rating Distribution (Value Counts):** The dataset is fairly balanced for high ratings (5 and 4), but also includes lower ratings:
    * Rating 5: 38
    * Rating 4: 37
    * Rating 2: 12
    * Rating 3: 12
    * Rating 1: 10

---
## 2. Prerequisites (Dependencies)
To execute the processing steps in the notebook, the following Python libraries are required:

* **`pandas`**: For data manipulation and DataFrame operations.
* **`nltk`**: The core library for most NLP tasks, including tokenization, stemming, lemmatization, and accessing stop words.
* **`re`**: Python's built-in module for regular expressions, used for pattern-based text cleaning.

---

## 3. Methodology and Steps

The processing pipeline is designed to transform raw review text into normalized, quantitative features suitable for machine learning.

1. Data Loading and Initial Analysis
* The data is loaded using `pd.read_csv('/content/hotel_reviews.csv')`.
* Initial checks confirm the size (`109 rows, 2 columns`) and column structure (`Review` and `Rating`).
* A check for missing values confirms **no missing data** exists in the dataset.

### 2. Text Cleaning
The text undergoes two stages of cleaning:
* **Stopword Removal:** Common English stopwords are removed from the `Review` text and stored in a new column, `clean_review`.
* **Punctuation Handling:**
    * The asterisk character (`*`) is specifically replaced with the word `star` (e.g., "4\* experience" becomes "4star experience").
    * All other remaining non-word characters are removed.

### 3. Text Normalization
* **Tokenization:** The cleaned text is broken down into individual word lists (tokens) using `nltk.word_tokenize` and saved in the `tokenize` column.
* **Stemming:** The **PorterStemmer** algorithm is applied to the tokens to reduce words to their base or root form (e.g., "stayed" to "stay").
* **Lemmatization:** The **WordNetLemmatizer** is used to reduce words to their dictionary or base form (lemma).

### 4. Feature Extraction (N-grams)
After normalization, the tokenized text is used to identify common phrases:
* All tokens are concatenated into a single list (`token_clean`).
* **Unigrams (1-word):** Calculated to find the most frequent single words.
* **Bigrams (2-words):** Calculated to find common two-word phrases, with results showing phrases like `great location` and `nice hotel`.
* **Trigrams (3-words):** Calculated to find common three-word phrases, such as `pike place market` and `hotel great location`.
