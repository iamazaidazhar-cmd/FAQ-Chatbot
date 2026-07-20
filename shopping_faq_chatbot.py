import streamlit as st
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
# -------------------- Page Configuration --------------------
 
st.set_page_config(
    page_title="ShopEase FAQ Chatbot",
    page_icon="🛒",
    layout="centered"
)
 
# -------------------- NLTK Setup (downloads run once, cached) --------------------
 
@st.cache_resource
def setup_nltk():
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    return True
 
setup_nltk()
 
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
 
# -------------------- FAQ Dataset --------------------
# Feel free to add, edit, or remove entries below.
 
FAQS = [
    {"question": "How do I place an order?",
     "answer": "Simply browse our products, add items to your cart, and click 'Checkout'. Follow the steps to enter your shipping and payment details to complete your order."},
    {"question": "What payment methods do you accept?",
     "answer": "We accept credit/debit cards, PayPal, Apple Pay, Google Pay, and cash on delivery in select areas."},
    {"question": "How can I track my order?",
     "answer": "Once your order ships, you'll receive a tracking link via email and SMS. You can also track it anytime from 'My Orders' in your account."},
    {"question": "What is your delivery time?",
     "answer": "Standard delivery takes 3-5 business days. Express delivery is available at checkout for 1-2 day delivery in most areas."},
    {"question": "Do you offer free shipping?",
     "answer": "Yes, we offer free standard shipping on all orders over $50. Orders below that have a flat shipping fee of $4.99."},
    {"question": "What is your return policy?",
     "answer": "You can return most items within 30 days of delivery for a full refund, as long as they're unused and in original packaging."},
    {"question": "How do I return or exchange an item?",
     "answer": "Go to 'My Orders', select the item, and click 'Return/Exchange'. Print the prepaid label and drop the package at any courier location."},
    {"question": "When will I get my refund?",
     "answer": "Refunds are processed within 5-7 business days after we receive and inspect the returned item."},
    {"question": "Can I cancel my order?",
     "answer": "Yes, you can cancel your order for free within 1 hour of placing it, as long as it hasn't been shipped yet, from the 'My Orders' page."},
    {"question": "Do you ship internationally?",
     "answer": "Yes, we ship to over 40 countries. International shipping costs and delivery times vary depending on the destination."},
    {"question": "How do I change my shipping address?",
     "answer": "You can update your shipping address before checkout, or contact customer support within 1 hour of ordering if you need to change it after placing an order."},
    {"question": "Is my payment information secure?",
     "answer": "Yes, we use industry-standard SSL encryption and never store your full card details on our servers."},
    {"question": "Do you have a loyalty or rewards program?",
     "answer": "Yes! Join ShopEase Rewards to earn points on every purchase, redeemable for discounts on future orders."},
    {"question": "How do I apply a discount or promo code?",
     "answer": "Enter your promo code in the 'Discount Code' box at checkout and click 'Apply' before completing your payment."},
    {"question": "What if I received a damaged or wrong item?",
     "answer": "We're sorry for the inconvenience! Contact our support team within 48 hours with photos of the item, and we'll arrange a free replacement or refund."},
    {"question": "Do you offer gift wrapping?",
     "answer": "Yes, gift wrapping is available for a small fee at checkout. You can also add a personalized gift message."},
    {"question": "How can I contact customer support?",
     "answer": "You can reach us via live chat on our website, email at support@shopease.com, or call us at +1-234-567-8900, available 9 AM to 9 PM daily."},
    {"question": "Do I need an account to place an order?",
     "answer": "No, you can checkout as a guest. However, creating a free account lets you track orders, save addresses, and earn reward points."},
    {"question": "Are the product prices inclusive of tax?",
     "answer": "Product prices displayed do not include tax. Applicable taxes are calculated and shown at checkout based on your shipping address."},
    {"question": "Can I change or add items to an order after placing it?",
     "answer": "Once an order is placed, items can't be added, but you can cancel within 1 hour (if not yet shipped) and place a new order."},
    {"question": "Do you have a mobile app?",
     "answer": "Yes! ShopEase is available on both the App Store and Google Play, offering the same features as our website plus exclusive app-only deals."},
    {"question": "What should I do if my order is delayed?",
     "answer": "If your order is past its estimated delivery date, check the tracking link first. If there's no update, contact our support team for assistance."},
    {"question": "What kind of products do you sell?",
     "answer": "We offer a wide range of products including electronics, fashion & apparel, home & kitchen essentials, beauty & personal care, books, and sports equipment."},
    {"question": "Do you sell branded or original products?",
     "answer": "Yes, all products sold on ShopEase are 100% genuine and sourced directly from authorized brands and suppliers."},
    {"question": "Do you have new arrivals or seasonal collections?",
     "answer": "Yes, we regularly update our catalog with new arrivals and seasonal collections. Check the 'New Arrivals' section on our homepage for the latest additions."},
]
 
# -------------------- Text Preprocessing --------------------
 
def preprocess(text: str) -> str:
    """Lowercase, remove punctuation/numbers, tokenize, remove stopwords, lemmatize."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in stop_words and tok not in string.punctuation and len(tok) > 1
    ]
    return " ".join(tokens)
 
 
@st.cache_resource
def build_vectorizer(_faqs):
    questions = [faq["question"] for faq in _faqs]
    processed_questions = [preprocess(q) for q in questions]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_questions)
    return vectorizer, tfidf_matrix
 
 
vectorizer, tfidf_matrix = build_vectorizer(FAQS)
 
SIMILARITY_THRESHOLD = 0.2
 
 
def get_best_answer(user_query: str):
    processed_query = preprocess(user_query)
    if not processed_query.strip():
        return "Could you please rephrase your question?", 0.0
 
    query_vector = vectorizer.transform([processed_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
 
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]
 
    if best_score < SIMILARITY_THRESHOLD:
        return ("Sorry, I couldn't find a matching answer for that. "
                "Try asking about orders, shipping, returns, payments, or your account!"), best_score
 
    return FAQS[best_idx]["answer"], best_score
 
 
# -------------------- CSS --------------------
 
def apply_theme(dark: bool):
    if dark:
        st.markdown("""
        <style>
        .stApp{ background:#0D1B1E; }
        h1,h2,h3,p,div,label,span{ color:#E6F1F0; }
        [data-testid="stSidebar"]{ background:#0A1315; }
        [data-testid="stSidebar"] *{ color:#E6F1F0; }
        .stChatMessage{ border-radius:12px; }
        [data-testid="stChatInput"] textarea{
            background:#132A2D !important;
            color:#E6F1F0 !important;
        }
        .stButton>button{
            background:#14B8A6; color:#0D1B1E; border-radius:8px; font-weight:bold;
        }
        .stButton>button:hover{ background:#0D9488; }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp{ background:#F0FDFA; }
        h1,h2,h3,p,div,label,span{ color:#134E4A; }
        [data-testid="stSidebar"]{ background:#134E4A; }
        [data-testid="stSidebar"] *{ color:#F0FDFA; }
        .stChatMessage{ border-radius:12px; }
        .stButton>button{
            background:#14B8A6; color:white; border-radius:8px; font-weight:bold;
        }
        .stButton>button:hover{ background:#0D9488; }
        </style>
        """, unsafe_allow_html=True)
 
# -------------------- Session State Init --------------------
 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hi there! 👋 Welcome to ShopEase Support. How can I help you today?"}
    ]
 
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
 
if "questions_asked" not in st.session_state:
    st.session_state.questions_asked = 0
 
# -------------------- Sidebar --------------------
 
st.sidebar.title("🛒 ShopEase")
st.sidebar.markdown("---")
 
# Settings
st.sidebar.subheader("⚙️ Settings")
dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=False)
show_confidence = st.sidebar.checkbox("📈 Show Match Confidence", value=False)
 
apply_theme(dark_mode)
 
st.sidebar.markdown("---")
 
# Quick Questions
st.sidebar.subheader("⚡ Quick Questions")
quick_questions = [
    "How do I track my order?",
    "What is your return policy?",
    "Do you offer free shipping?",
    "How do I cancel my order?",
]
for q in quick_questions:
    if st.sidebar.button(q, use_container_width=True):
        st.session_state.pending_query = q
 
st.sidebar.markdown("---")
 
# Browse all FAQs
with st.sidebar.expander("📋 Browse All FAQs"):
    for faq in FAQS:
        st.markdown(f"**Q: {faq['question']}**")
        st.caption(faq["answer"])
        st.markdown("—")
 
st.sidebar.markdown("---")
 
# About
with st.sidebar.expander("ℹ️ About"):
    st.write("This chatbot matches your questions to our FAQ database using "
             "NLP preprocessing (NLTK) and TF-IDF + Cosine Similarity.")
    st.write("Version: 1.0")
 
st.sidebar.caption("Developed by Muhammad Zaid Azhar")
 
# -------------------- Title --------------------
 
st.title("🛒 ShopEase FAQ Chatbot")
st.write("Ask me anything about your orders, shipping, returns, and more!")
 
# -------------------- Statistics --------------------
 
st.subheader("📊 Statistics")
stat_col1, stat_col2 = st.columns(2)
stat_col1.metric("📚 FAQs Loaded", len(FAQS))
stat_col2.metric("💬 Questions Asked", st.session_state.questions_asked)
 
st.markdown("---")
 
# -------------------- Quick Actions --------------------
 
st.subheader("🛠️ Quick Actions")
qa_col1, qa_col2 = st.columns(2)
 
with qa_col1:
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hi there! 👋 Welcome to ShopEase Support. How can I help you today?"}
        ]
        st.session_state.questions_asked = 0
        st.rerun()
 
with qa_col2:
    chat_text = "\n\n".join(
        f"{'You' if m['role'] == 'user' else 'Bot'}: {m['content']}"
        for m in st.session_state.chat_history
    )
    st.download_button(
        "📥 Download Chat History",
        data=chat_text,
        file_name="shopease_chat.txt",
        mime="text/plain",
        use_container_width=True
    )
 
st.markdown("---")
 
# -------------------- Display Chat History --------------------
 
def render_assistant_message(content: str):
    st.write(content)
    with st.expander("📋 Copy Answer"):
        st.code(content, language=None)
 
 
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_assistant_message(msg["content"])
        else:
            st.write(msg["content"])
 
# -------------------- Chat Input --------------------
 
typed_input = st.chat_input("Type your question here...")
 
# Use quick-question click if present, otherwise use typed input
user_input = st.session_state.pending_query or typed_input
st.session_state.pending_query = None
 
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.questions_asked += 1
    with st.chat_message("user"):
        st.write(user_input)
 
    answer, score = get_best_answer(user_input)
    display_answer = answer
    if show_confidence:
        display_answer += f"\n\n*(Match confidence: {score:.0%})*"
 
    with st.chat_message("assistant"):
        render_assistant_message(display_answer)
 
    st.session_state.chat_history.append({"role": "assistant", "content": display_answer})
 
# -------------------- Footer --------------------
 
st.markdown("---")
st.caption("Developed by Muhammad Zaid Azhar")