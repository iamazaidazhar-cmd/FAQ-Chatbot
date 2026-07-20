# 🛒 ShopEase FAQ Chatbot

A simple NLP-powered FAQ chatbot built with **Streamlit**, **NLTK**, and **scikit-learn**. It matches user questions to a predefined FAQ database using TF-IDF vectorization and cosine similarity.

## Features

- 💬 Interactive chat interface for asking shopping-related questions
- 🔍 NLP preprocessing (lowercasing, stopword removal, lemmatization) via NLTK
- 📊 TF-IDF + Cosine Similarity matching against 25+ FAQ entries
- ⚡ Quick question buttons in the sidebar
- 📋 Browse all FAQs directly from the sidebar
- 🌙 Light/Dark mode toggle
- 📈 Optional match confidence display
- 📥 Download chat history as a text file
- 🗑 Clear chat button
- 📊 Live stats (FAQs loaded, questions asked)

## Requirements

- Python 3.8+
- streamlit
- nltk
- scikit-learn

## Installation

1. **Clone or download this project**

2. **Create and activate a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements_shopping_faq_chatbot.txt
   ```

   > NLTK data packages (`punkt`, `punkt_tab`, `stopwords`, `wordnet`, `omw-1.4`) are downloaded automatically on first run — no manual setup needed.

## Usage

Run the app with Streamlit:

```bash
streamlit run shopping_faq_chatbot.py
```

Then open the local URL shown in your terminal (usually `http://localhost:8501`).

## How It Works

1. All FAQ questions are preprocessed (lowercased, cleaned, tokenized, stopwords removed, lemmatized).
2. A **TF-IDF vectorizer** is fitted on the preprocessed FAQ questions.
3. When a user asks a question, it's preprocessed the same way and converted into a TF-IDF vector.
4. **Cosine similarity** is computed between the user's query and all FAQ questions.
5. The best-matching FAQ answer is returned — if the similarity score is below a threshold (`0.2`), the bot asks the user to rephrase or suggests topics instead.

## Project Structure

```
.
├── shopping_faq_chatbot.py               # Main Streamlit app
├── requirements_shopping_faq_chatbot.txt # Python dependencies
└── README.md                             # This file
```

## Customization

- **Add/edit FAQs:** modify the `FAQS` list near the top of `shopping_faq_chatbot.py`.
- **Adjust match sensitivity:** change `SIMILARITY_THRESHOLD` (default `0.2`) — lower values make the bot more lenient, higher values make it stricter.
- **Theming:** edit the CSS inside the `apply_theme()` function.

## Author

Developed by **Muhammad Zaid Azhar**
