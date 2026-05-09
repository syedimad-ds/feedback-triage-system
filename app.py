import streamlit as st
import pandas as pd
import os
import json
import time
from datetime import datetime
import concurrent.futures
from crewai import Agent, Task, Crew, Process

# Set page config
st.set_page_config(page_title="Feedback Analysis System", layout="wide")

# Disable CrewAI Telemetry (prevents hanging in Colab/Streamlit)
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

st.title("🤖 Intelligent User Feedback Analysis System")
st.write("Streamlit UI + Fast CrewAI Orchestration (Colab Integrated)")

# 1. Try environment variable (Colab)
api_key = os.environ.get("GEMINI_API_KEY")

# 2. Try Streamlit Secrets (Streamlit Share)
if not api_key:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
        elif "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass

# Always show the sidebar input just in case they want to override or test
api_key_input = st.sidebar.text_input("Enter Gemini API Key (Overrides Secret/Env)", type="password")
if api_key_input:
    api_key = api_key_input

if api_key:
    os.environ["GEMINI_API_KEY"] = api_key
else:
    st.warning("⚠️ API Key not found. Please provide it in the sidebar, Streamlit Secrets, or Colab environment.")
    st.stop()

def generate_mock_data():
    reviews_data = [
        {"review_id": "R001", "platform": "Google Play", "rating": 1, "review_text": "App crashes when I try to sync my data.", "user_name": "user123", "date": "2023-10-01", "app_version": "2.1.3"},
        {"review_id": "R002", "platform": "App Store", "rating": 4, "review_text": "Please add a dark mode, would love to see it.", "user_name": "nightowl", "date": "2023-10-02", "app_version": "3.0.1"},
        {"review_id": "R003", "platform": "Google Play", "rating": 5, "review_text": "Amazing app! Works perfectly for my daily tasks.", "user_name": "pro_user", "date": "2023-10-03", "app_version": "3.0.1"},
        {"review_id": "R004", "platform": "App Store", "rating": 2, "review_text": "Too expensive for what it does. Also customer service is poor.", "user_name": "unhappy_cust", "date": "2023-10-04", "app_version": "2.1.3"},
        {"review_id": "R005", "platform": "Google Play", "rating": 1, "review_text": "Buy cheap watches here http://spam.link", "user_name": "spambot", "date": "2023-10-05", "app_version": "1.0.0"}
    ]
    pd.DataFrame(reviews_data).to_csv("app_store_reviews.csv", index=False)

    emails_data = [
        {"email_id": "E001", "subject": "App Crash Report", "body": "Hi, I am using an iPhone 13 on iOS 16. The app crashes immediately after logging in. Please fix.", "sender_email": "john@example.com", "timestamp": "2023-10-01T10:00:00Z", "priority": "High"},
        {"email_id": "E002", "subject": "Feature Request: Export to PDF", "body": "It would be great if I could export my weekly reports to PDF.", "sender_email": "jane@example.com", "timestamp": "2023-10-02T14:30:00Z", "priority": ""},
        {"email_id": "E003", "subject": "Login Issue", "body": "I can't login since the latest update. It says invalid credentials but I reset my password.", "sender_email": "bob@example.com", "timestamp": "2023-10-03T09:15:00Z", "priority": "High"}
    ]
    pd.DataFrame(emails_data).to_csv("support_emails.csv", index=False)
    
    expected_data = [
        {"source_id": "R001", "source_type": "Review", "category": "Bug", "priority": "Critical", "technical_details": "Crash on data sync, v2.1.3", "suggested_title": "Fix crash during data sync"},
        {"source_id": "R002", "source_type": "Review", "category": "Feature Request", "priority": "Medium", "technical_details": "N/A", "suggested_title": "Implement Dark Mode"},
        {"source_id": "E001", "source_type": "Email", "category": "Bug", "priority": "Critical", "technical_details": "iPhone 13, iOS 16, Crash on login", "suggested_title": "Fix immediate crash on login for iOS 16"}
    ]
    pd.DataFrame(expected_data).to_csv("expected_classifications.csv", index=False)

@st.cache_data
def load_data():
    if not os.path.exists("app_store_reviews.csv") or not os.path.exists("support_emails.csv") or not os.path.exists("expected_classifications.csv"):
        generate_mock_data()
    return pd.read_csv("app_store_reviews.csv"), pd.read_csv("support_emails.csv"), pd.read_csv("expected_classifications.csv")

reviews_df, emails_df, expected_df = load_data()

st.sidebar.header("Configuration")
process_limit = st.sidebar.slider("Number of items to process", 1, len(reviews_df) + len(emails_df), 5)

def get_items_to_process():
    items = []
    for _, row in reviews_df.iterrows():
        items.append({
            'source_id': str(row.get('review_id', '')),
            'source_type': 'app_store_review',
            'platform': str(row.get('platform', '')),
            'rating': row.get('rating', None),
            'text': str(row.get('review_text', '')),
            'subject': '',
            'app_version': str(row.get('app_version', ''))
        })
    for _, row in emails_df.iterrows():
        items.append({
            'source_id': str(row.get('email_id', '')),
            'source_type': 'support_email',
            'platform': '',
            'rating': None,
            'text': str(row.get('body', '')),
            'subject': str(row.get('subject', '')),
            'app_version': ''
        })
    return items[:process_limit]

# Define LLM Model globally for speed (Flash is the fastest architecture)
LLM_MODEL_STRING = "gemini/gemini-1.5-flash"

def process_single_item(item):
    """Processes a single item through CrewAI. Designed to be run in parallel."""
    try:
        # AGENTS (Kept lightweight and specific to the task)
        classifier = Agent(
            role='Feedback Classifier',
            goal='Categorize feedback into one category: Bug, Feature Request, Praise, Complaint, or Spam.',
            backstory='Expert in reading user feedback.',
            verbose=False, allow_delegation=False, llm=LLM_MODEL_STRING
        )
        
        analyzer = Agent(
            role='Technical Analyzer',
            goal='Extract technical details and priority.',
            backstory='Technical support engineer extracting actionable bugs or feature specifics.',
            verbose=False, allow_delegation=False, llm=LLM_MODEL_STRING
        )
        
        ticket_creator = Agent(
            role='Ticket Creator',
            goal='Draft a structured ticket with a strict JSON format.',
            backstory='Product manager translating notes to tickets.',
            verbose=False, allow_delegation=False, llm=LLM_MODEL_STRING
        )
        
        quality_critic = Agent(
            role='Quality Critic',
            goal='Review drafted tickets for complete JSON output.',
            backstory='QA engineer strictly ensuring JSON validity.',
            verbose=False, allow_delegation=False, llm=LLM_MODEL_STRING
        )

        # TASKS
        t1 = Task(
            description=f"Classify strictly into Bug, Feature Request, Praise, Complaint, Spam: '{item['text']}'. Respond with valid JSON: {{\"category\": \"<Cat>\"}}",
            expected_output="JSON classification",
            agent=classifier
        )
        
        t2 = Task(
            description=f"Extract technical info (device, OS, steps, severity) from: '{item['text']}'. Respond with valid JSON.",
            expected_output="JSON technical details",
            agent=analyzer
        )
        
        t3 = Task(
            description=f"Using the analysis, draft an engineering ticket. Respond with valid JSON: {{\"title\": \"<tag> <title>\", \"description\": \"<desc>\", \"priority\": \"<pri>\"}}",
            expected_output="JSON ticket draft",
            agent=ticket_creator
        )

        t4 = Task(
            description="Fix any formatting errors in the ticket and output the FINAL valid JSON: {\"category\": \"...\", \"priority\": \"...\", \"title\": \"...\", \"description\": \"...\", \"quality_score\": 9}. Strictly JSON only, no markdown formatting.",
            expected_output="Final JSON string",
            agent=quality_critic
        )
        
        crew = Crew(
            agents=[classifier, analyzer, ticket_creator, quality_critic],
            tasks=[t1, t2, t3, t4],
            verbose=False, # Disable verbose to speed up logs
            process=Process.sequential
        )
        
        res = crew.kickoff()
        res_str = str(res.raw if hasattr(res, 'raw') else res)
        
        import re
        json_match = re.search(r'\{.*\}', res_str, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
        else:
            parsed = json.loads(res_str)
            
        parsed['source_id'] = item['source_id']
        parsed['source_type'] = item['source_type']
        return parsed
        
    except Exception as e:
        return {"source_id": item['source_id'], "category": "Error", "title": f"Failed: {str(e)}"}


if st.button("🚀 Process Feedback (Sequential for Stability)"):
    st.write("### Running CrewAI Agents...")
    items = get_items_to_process()
    progress_bar = st.progress(0)
    
    processed_data = []
    
    start_time = time.time()
    
    for i, item in enumerate(items):
        with st.spinner(f"Analyzing {item['source_id']} ({i+1}/{len(items)})..."):
            result = process_single_item(item)
            processed_data.append(result)
            progress_bar.progress((i + 1) / len(items))

    elapsed = time.time() - start_time
    st.success(f"⚡ Pipeline Complete in {elapsed:.2f} seconds!")
    
    # Display Results
    res_df = pd.DataFrame(processed_data)
    st.write("### Generated Tickets")
    st.dataframe(res_df)
    res_df.to_csv("generated_tickets.csv", index=False)
    
    # Accuracy Check vs Expected
    st.write("### 📊 Accuracy Evaluation")
    merged = res_df.merge(expected_df[['source_id', 'category', 'priority']].rename(columns={'category': 'expected_category', 'priority': 'expected_priority'}), on='source_id', how='inner')
    if not merged.empty and 'category' in merged.columns:
        cat_correct = (merged['category'].str.lower() == merged['expected_category'].str.lower()).sum()
        st.metric("Category Accuracy", f"{cat_correct}/{len(merged)} ({100*cat_correct/len(merged):.1f}%)")
        st.dataframe(merged[['source_id', 'category', 'expected_category', 'title']])
        
st.write("---")
st.write("### Manual Override / Generated Tickets View")
if os.path.exists("generated_tickets.csv"):
    df_tickets = pd.read_csv("generated_tickets.csv")
    edited_df = st.data_editor(df_tickets)
    if st.button("Save Changes"):
        edited_df.to_csv("generated_tickets.csv", index=False)
        st.success("Changes saved successfully.")
