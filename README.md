# Consumer Complaints Topic Analysis

This project applies Natural Language Processing (NLP) techniques to identify prevalent topics in consumer financial complaints. Two different topic-modelling pipelines are implemented and compared:

1. **TF-IDF + Non-negative Matrix Factorization (NMF)**
2. **Sentence-BERT + BERTopic**

The goal is to compare a traditional term-frequency-based approach with a modern semantic embedding approach and evaluate their topic specificity, interpretability, and clustering behaviour.

## Dataset

The project uses a consumer complaints dataset obtained from Kaggle and based on consumer financial complaint data. The original dataset contained **162,421 records** with complaint narratives and product categories.

During preprocessing:

- the unnecessary index column was removed;
- missing narratives were removed;
- duplicate complaint narratives were removed;
- very short narratives containing fewer than five words were removed;
- a reproducible stratified sample of **10,000 complaints** was created using `random_state=42`.

The same 10,000 complaints were used for both pipelines to ensure a fair comparison.

The dataset itself is not stored in this repository.

Dataset source:

Consumer Complaints Dataset for NLP – Kaggle  
https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp

## Project Structure

### `01_data_exploration_preprocessing.ipynb`

Loads and explores the original dataset, checks missing values and duplicates, analyses category distributions and complaint lengths, verifies the existing preprocessing, removes unsuitable records, and creates the final 10,000-row stratified sample.

### `02_tfidf_nmf.ipynb`

Implements the traditional NLP pipeline:

**Complaint text → TF-IDF vectors → NMF topic modelling**

TF-IDF uses a maximum vocabulary of 5,000 features with unigrams and bigrams. NMF extracts eight interpretable topics.

### `03_sbert_bertopic.ipynb`

Implements the semantic NLP pipeline:

**Complaint text → Sentence-BERT embeddings → BERTopic**

The `all-MiniLM-L6-v2` Sentence-BERT model converts each complaint into a 384-dimensional semantic embedding. BERTopic then uses UMAP and HDBSCAN to discover semantic clusters.

### `04_results_comparison.ipynb`

Compares the two topic-modelling pipelines using topic prevalence, topic mappings, normalized cross-tabulation, visualization, and Normalized Mutual Information (NMI).

## Main Results

### TF-IDF + NMF

NMF produced eight relatively specific topics:

- Banking Customer Service / Funds Issues
- Credit Reporting Disputes
- Loan & Mortgage Payment Issues
- Bank Account Opening / Closure
- Debt Collection & Validation
- Credit Card & Balance Issues
- Identity Theft & Fraud
- Hard Credit Inquiries

The most prevalent NMF topic was **Banking Customer Service / Funds Issues (24.19%)**.

### Sentence-BERT + BERTopic

BERTopic produced four main semantic topics plus an outlier group:

- Credit Reporting & Account Disputes
- Banking, Loan & Payment Issues
- Digital Payments, Transactions & Refunds
- Bank Account Linking Issues
- Outliers / Unassigned

The largest BERTopic cluster was **Credit Reporting & Account Disputes (50.19%)**.

## Comparison

The two approaches produced different topic structures.

TF-IDF + NMF generated more detailed and operationally specific topics. BERTopic grouped semantically related complaints into fewer and broader themes.

For example, NMF topics such as Credit Reporting Disputes, Debt Collection & Validation, Identity Theft & Fraud, and Hard Credit Inquiries were mainly grouped within BERTopic's broader **Credit Reporting & Account Disputes** cluster.

The Normalized Mutual Information score between the two clustering results was **0.219**, indicating limited agreement between the two topic structures.

For this dataset, TF-IDF + NMF provides greater topic specificity, while Sentence-BERT + BERTopic provides a broader semantic view of the complaints.

## Requirements

The notebooks were developed using Python in Google Colab.

Main libraries:

- pandas
- NumPy
- matplotlib
- scikit-learn
- sentence-transformers
- BERTopic
- UMAP
- HDBSCAN

## Running the Project

Run the notebooks in the following order:

1. `01_data_exploration_preprocessing.ipynb`
2. `02_tfidf_nmf.ipynb`
3. `03_sbert_bertopic.ipynb`
4. `04_results_comparison.ipynb`

The original complaint CSV is required for Notebook 1. Notebook 1 creates the 10,000-row sample used by both topic-modelling pipelines.


The required packages for the BERTopic pipeline can be installed with:

```bash
pip install bertopic sentence-transformers umap-learn hdbscan
```

## NMF Topic Number Selection

The number of NMF topics was evaluated systematically instead of being chosen arbitrarily. Models containing between 5 and 12 topics were compared using reconstruction error, mean topic similarity, topic diversity, NPMI coherence, and manual interpretability.

The eight-topic solution achieved an NPMI coherence score of **0.2541**, clearly improving on the 5–7 topic solutions and remaining close to the highest observed score of **0.2590** for 10 topics.

Although models with more topics produced slightly better numerical scores, manual inspection showed increasing topic fragmentation. For example, the 10-topic solution introduced a narrow Wells Fargo-specific topic containing only 191 complaints.

Therefore, **8 topics were selected as the optimal practical solution**, providing a strong balance between coherence, interpretability, topic distinctiveness, sufficient topic size, and model parsimony.

## Final Eight NMF Topics

| Topic | Label | Main Keywords |
|---|---|---|
| 1 | Hard Credit Inquiries | inquiry, inquiry credit, credit, hard inquiry, report, credit report, unauthorized |
| 2 | Banking Customer Service / Funds Issues | bank, call, told, called, money, back, said, time |
| 3 | Loan & Mortgage Payment Issues | payment, late, loan, mortgage, late payment, month, due, interest |
| 4 | Debt Collection & Validation | debt, collection, company, agency, collection agency, validation, debt collection |
| 5 | Bank Account Opening / Closure | account, opened, balance, closed, bank, account number, checking |
| 6 | Credit Card & Balance Issues | card, credit card, credit, charge, capital one, balance, credit score |
| 7 | Identity Theft & Fraud | identity, theft, identity theft, fraudulent, victim, fraudulent account |
| 8 | Credit Reporting Disputes | credit, report, credit report, information, reporting, bureau, experian, dispute, inaccurate |
