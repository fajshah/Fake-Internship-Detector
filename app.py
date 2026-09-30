import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="InternGuard AI - WTQ 2026", page_icon="🛡️")
st.title("🛡️ InternGuard AI")
st.write("Fake Internship Detector - Built for Women Tech Quest 2026")
st.write("Built by Syeda Farzana Shah")

@st.cache_resource
def train_model():
    data = [
        ("Live Training. Remote Flexibility. Verified Certificate. Small nominal fee required for learning portal", 1),
        ("Paid internship at 10Pearls, stipend 50k, interview process, no fee", 0),
        ("Earn 1 lac per month, just pay 2000 registration fee, WhatsApp us", 1),
        ("True Classic Tees hiring via hr@trueclassictees.info, send CV", 1),
        ("Microsoft Learn Student Ambassador program, learn Copilot, free", 0),
        ("Google Summer of Code, open source, stipend $3000", 0),
        ("Data Science internship, work on real projects, mentor assigned, no fee", 0),
        ("Marketing ads specialist, send your bank details for payroll setup", 1),
        ("Build with AI competition, 3 hour challenge, bring laptop", 0),
        ("Free access to Zidio learning + job portal after paying training fee", 1),
        ("Internship with nominal fee of 1200 for certificate and portal access", 1),
        ("Internship with no fee, paid stipend, official company email", 0),
        ("Urgent hiring! Pay 5000 and get offer letter today", 1),
        ("Software Engineer Intern at Systems Ltd, Karachi, proper interview", 0),
    ]
    df = pd.DataFrame(data, columns=["description", "is_fake"])
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['description'])
    model = LogisticRegression()
    model.fit(X, df['is_fake'])
    return model, vectorizer

model, vectorizer = train_model()

desc = st.text_area("Internship ka description yahan paste karo:")

if st.button("Check Karo"):
    if desc:
        X_test = vectorizer.transform([desc])
        pred = model.predict(X_test)[0]
        prob = model.predict_proba(X_test)[0]
        conf = max(prob)*100

        if pred == 1:
            st.error(f"❌ FAKE HAI! ({conf:.1f}% confidence)")
            st.write("Wajah:")
            if "fee" in desc.lower() or "pay" in desc.lower() or "1200" in desc.lower():
                st.write("• 'Fee' mang raha hai - Real companies kabhi fee nahi mangti")
            if "whatsapp" in desc.lower():
                st.write("• WhatsApp par hiring - Professional nahi hai")
        else:
            st.success(f"✅ REAL lag raha hai ({conf:.1f}% confidence)")
    else:
        st.warning("Pehle description likho!")