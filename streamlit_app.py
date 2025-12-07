# streamlit_app.py
"""
Streamlit app for Task 2 — Two-Dashboard AI Feedback System
User Dashboard (public): submit rating + review -> AI reply -> saved
Admin Dashboard (internal): live list of submissions, AI summaries, recommended actions + analytics

Storage: CSV file in app directory ('submissions.csv')
LLM: Perplexity sonar-pro via REST (requests). Set API key in STREAMLIT secrets or env var PERPLEXITY_API_KEY.
"""

import streamlit as st
import pandas as pd
import requests
import json
import os
import re
from datetime import datetime
from io import StringIO

# ---------- CONFIG ----------
API_URL = "https://api.perplexity.ai/chat/completions"
MODEL = "sonar-pro"
STORAGE_FILE = "submissions.csv"
# ----------------------------

st.set_page_config(layout="wide", page_title="AI Feedback System")

# ---------- Helpers: LLM call & JSON extractor ----------
def call_perplexity(prompt, api_key, max_tokens=500, temperature=0.0):
    """Call Perplexity sonar-pro; returns raw text or None."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        # "temperature": temperature # if supported by endpoint; optional
    }
    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if r.status_code == 200:
            data = r.json()
            return data["choices"][0]["message"]["content"]
        else:
            st.error(f"LLM API error {r.status_code}: {r.text[:300]}")
            return None
    except Exception as e:
        st.error(f"LLM call exception: {e}")
        return None

def extract_json_fragment(text):
    """Return parsed JSON from within a text blob, or None."""
    if text is None: 
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except:
        return None

# ---------- Storage helpers ----------
def ensure_storage():
    try:
        df = pd.read_csv(STORAGE_FILE)
    except FileNotFoundError:
        df = pd.DataFrame(columns=[
            "id","timestamp","rating","review","ai_reply","ai_summary","ai_actions"
        ])
        df.to_csv(STORAGE_FILE, index=False)
    return

def load_submissions():
    try:
        df = pd.read_csv(STORAGE_FILE)
    except FileNotFoundError:
        df = pd.DataFrame(columns=[
            "id","timestamp","rating","review","ai_reply","ai_summary","ai_actions"
        ])
    return df

def append_submission(row: dict):
    df = load_submissions()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(STORAGE_FILE, index=False)

def update_submission(index, updates: dict):
    df = load_submissions()
    for k,v in updates.items():
        df.at[index, k] = v
    df.to_csv(STORAGE_FILE, index=False)

import os

def get_api_key():
    # First: environment variable
    key = os.getenv("PERPLEXITY_API_KEY")
    if key:
        return key
    
    # Second: Streamlit secrets (only available on deployment)
    try:
        return st.secrets["PERPLEXITY_API_KEY"]
    except:
        return None

# ---------- Prompts ----------
PROMPT_USER_REPLY = """You are a helpful customer-response assistant.
Given the user's rating and review, respond politely and helpfully in a single short paragraph.

Rules:
- Output exactly one short reply (no JSON).
- Keep it <= 2 sentences.
- Tone: friendly and constructive.
- If rating <= 2, apologize and offer to escalate; if rating >= 4, thank and suggest next positive step.

Input:
Rating: {rating}
Review: "{review}"
"""

PROMPT_ADMIN_SUMMARY = """You are a concise summariser for admins.
Given a user's review, provide a one-line summary and a single short recommended action for internal teams.

Return JSON ONLY in this format:
{{ "summary": "<one-line summary>", "recommended_action": "<one short action>" }}

Review:
"{review}"
Rating: {rating}
"""

# ---------- UI: Tabs for User and Admin ----------
ensure_storage()
tab1, tab2 = st.tabs(["User Dashboard", "Admin Dashboard"])

# ---------- USER DASHBOARD ----------
with tab1:
    st.header("User Dashboard — Submit a Review")
    st.markdown("Select rating, write a short review, submit — an AI reply will be generated and saved.")
    col1, col2 = st.columns([3,1])

    with col1:
        rating = st.selectbox("Star rating", [5,4,3,2,1], index=0)
        review_text = st.text_area("Write your review", height=140, placeholder="Write a short review...")
        submit_btn = st.button("Submit review")

    with col2:
        st.markdown("*Quick tips*")
        st.write("- Be concise (1–3 sentences).")
        st.write("- Mention what you liked or disliked.")
        st.write("- The AI reply will be short and friendly.")

    if submit_btn:
        if not review_text.strip():
            st.warning("Please write a review before submitting.")
        else:
            api_key = get_api_key()
            if not api_key:
                st.error("Perplexity API key required. Add to Streamlit secrets or paste it in the sidebar input.")
            else:
                st.info("Generating AI reply...")
                prompt = PROMPT_USER_REPLY.format(rating=rating, review=review_text.replace('"','\\"'))
                ai_reply = call_perplexity(prompt, api_key) or ""
                # Build row and save
                now = datetime.utcnow().isoformat()
                df = load_submissions()
                next_id = int(df["id"].max())+1 if (not df.empty and pd.notnull(df["id"].max())) else 1
                row = {
                    "id": next_id,
                    "timestamp": now,
                    "rating": rating,
                    "review": review_text,
                    "ai_reply": ai_reply,
                    "ai_summary": "",
                    "ai_actions": ""
                }
                append_submission(row)
                st.success("Saved! AI reply below:")
                st.write(ai_reply)

# ---------- ADMIN DASHBOARD ----------
with tab2:
    st.header("Admin Dashboard — Submissions & Actions")
    st.markdown("This internal view shows all submissions and lets admins generate summaries and recommended actions.")
    st.sidebar.markdown("Admin Controls")
    api_key = get_api_key()

    refresh_btn = st.button("Refresh submissions")
    regenerate_btn = st.button("Regenerate summaries for missing items")
    download_btn = st.button("Download CSV")

    df = load_submissions()
    st.write(f"Total submissions: {len(df)}")

    # Show basic analytics
    if not df.empty:
        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Avg rating", round(df['rating'].dropna().astype(float).mean(),2))
        with colB:
            counts = df['rating'].value_counts().to_dict()
            st.write("Ratings count:", counts)
        with colC:
            st.write("Last 5 reviews:")
            st.table(df.sort_values("timestamp", ascending=False)[["timestamp","rating","review"]].head(5))

    # Table + per-row actions
    st.markdown("---")
    st.markdown("### Submissions")
    if df.empty:
        st.info("No submissions yet.")
    else:
        # editable view: admin can request summary for any row
        for idx, row in df.iterrows():
            st.markdown(f"*ID {int(row['id'])} — Rating: {int(row['rating'])} — {row['timestamp']}*")
            st.write(row['review'])
            st.write("AI Reply:", row.get('ai_reply', ''))
            cols = st.columns([1,1,1,4])
            with cols[0]:
                if st.button(f"Summarize #{int(row['id'])}", key=f"summ_{idx}"):
                    if not api_key:
                        st.error("Perplexity API key required to summarize.")
                    else:
                        # call summarizer
                        prompt = PROMPT_ADMIN_SUMMARY.format(review=row['review'].replace('"','\\"'), rating=int(row['rating']))
                        out = call_perplexity(prompt, api_key)
                        parsed = extract_json_fragment(out)
                        if parsed:
                            update_submission(idx, {"ai_summary": parsed.get("summary",""), "ai_actions": parsed.get("recommended_action","")})
                            st.success("Summary saved.")
                        else:
                            st.warning("Could not parse JSON from LLM response. Raw output shown below.")
                            st.write(out)

            with cols[1]:
                if st.button(f"Regenerate reply #{int(row['id'])}", key=f"reply_{idx}"):
                    if not api_key:
                        st.error("Perplexity API key required.")
                    else:
                        prompt = PROMPT_USER_REPLY.format(rating=int(row['rating']), review=row['review'].replace('"','\\"'))
                        out = call_perplexity(prompt, api_key)
                        update_submission(idx, {"ai_reply": out})
                        st.success("AI reply regenerated and saved.")

            with cols[2]:
                if st.button(f"Delete #{int(row['id'])}", key=f"del_{idx}"):
                    df2 = df.drop(index=idx).reset_index(drop=True)
                    df2.to_csv(STORAGE_FILE, index=False)
                    st.experimental_rerun()

            with cols[3]:
                st.write("Summary:", row.get("ai_summary",""))
                st.write("Action:", row.get("ai_actions",""))

            st.markdown("---")

    # Bulk actions
    if regenerate_btn:
        if not api_key:
            st.error("Perplexity API key required to run regenerate.")
        else:
            st.info("Regenerating summaries for items with missing summaries...")
            df = load_submissions()
            for idx, row in df.iterrows():
                if not row.get("ai_summary"):
                    prompt = PROMPT_ADMIN_SUMMARY.format(review=row['review'].replace('"','\\"'), rating=int(row['rating']))
                    out = call_perplexity(prompt, api_key)
                    parsed = extract_json_fragment(out)
                    if parsed:
                        update_submission(idx, {
                            "ai_summary": parsed.get("summary",""),
                            "ai_actions": parsed.get("recommended_action","")
                        })
            st.success("Done. Refresh to see updates.")

    if download_btn:
        df = load_submissions()
        csv = df.to_csv(index=False)
        st.download_button("Download CSV", data=csv, file_name="submissions.csv", mime="text/csv")
