import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
from crewai import Agent, Task, Crew, Process

# Set page config
st.set_page_config(page_title="Feedback Analysis System", layout="wide")

st.title("Intelligent User Feedback Analysis System")
st.write("This application uses multiple AI agents to automatically read, categorize, and process user feedback into actionable tickets.")

# Anyone using the deployed app can put their own API Key
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password", help="Get your API key from Google AI Studio")

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

@st.cache_data
def load_data():
    if not os.path.exists("app_store_reviews.csv") or not os.path.exists("support_emails.csv"):
        generate_mock_data()
    return pd.read_csv("app_store_reviews.csv"), pd.read_csv("support_emails.csv")

reviews_df, emails_df = load_data()

st.sidebar.header("Configuration")
process_limit = st.sidebar.slider("Number of items to process", 1, 10, 3)

def process_feedback(feedback_text, source_type):
    # CrewAI (via litellm) natively supports "gemini/..." directly without the LangChain wrapper.
    llm_string = "gemini/gemini-1.5-flash"
    
    classifier = Agent(
        role='Feedback Classifier',
        goal='Categorize the feedback into Bug, Feature Request, Praise, Complaint, or Spam.',
        backstory='Expert in reading user feedback and identifying the core intent.',
        verbose=True,
        allow_delegation=False,
        llm=llm_string
    )
    
    analyzer = Agent(
        role='Technical Analyzer',
        goal='Extract technical details and determine priority (Critical, High, Medium, Low).',
        backstory='A seasoned QA engineer who can extract actionable reproduction steps and device info.',
        verbose=True,
        allow_delegation=False,
        llm=llm_string
    )
    
    ticket_creator = Agent(
        role='Ticket Creator',
        goal='Generate a structured ticket representation in JSON format.',
        backstory='An organized product manager who writes clear and concise tickets.',
        verbose=True,
        allow_delegation=False,
        llm=llm_string
    )
    
    quality_critic = Agent(
        role='Quality Critic',
        goal='Review generated tickets for completeness and accuracy, outputting final JSON.',
        backstory='A strict QA lead who ensures all tickets meet formatting standards.',
        verbose=True,
        allow_delegation=False,
        llm=llm_string
    )

    t1 = Task(
        description=f"Analyze the following {source_type} feedback: '{feedback_text}'. Classify it strictly into one of: Bug, Feature Request, Praise, Complaint, Spam.",
        expected_output="A single word or short phrase representing the category.",
        agent=classifier
    )
    
    t2 = Task(
        description=f"Based on the feedback: '{feedback_text}', extract technical details (device, OS, version, steps) and assign a priority (Critical, High, Medium, Low).",
        expected_output="A summary of technical details and the assigned priority.",
        agent=analyzer
    )
    
    t3 = Task(
        description="Using the category and technical details, create a ticket. Provide the ticket information in text format including category, priority, technical_details, suggested_title, and description.",
        expected_output="Draft ticket text.",
        agent=ticket_creator
    )

    t4 = Task(
        description="Review the drafted ticket. Fix any errors and strictly output raw JSON with keys: 'category', 'priority', 'technical_details', 'suggested_title', 'description'. Do not use markdown blocks, just the JSON.",
        expected_output="A valid JSON string strictly containing the keys: category, priority, technical_details, suggested_title, description.",
        agent=quality_critic
    )
    
    crew = Crew(
        agents=[classifier, analyzer, ticket_creator, quality_critic],
        tasks=[t1, t2, t3, t4],
        verbose=True,
        process=Process.sequential
    )
    
    return crew.kickoff()

if st.button("Process Feedback"):
    if not api_key:
        st.error("Please provide a Gemini API Key in the sidebar.")
    else:
        # CrewAI reads from the GEMINI_API_KEY environment variable.
        os.environ["GEMINI_API_KEY"] = api_key
        
        st.write("### Processing...")
        items_to_process = []
        for _, row in reviews_df.iterrows():
            items_to_process.append({"id": row['review_id'], "type": "Review", "text": f"Rating: {row['rating']}, Text: {row['review_text']}, Version: {row['app_version']}"})
        
        for _, row in emails_df.iterrows():
            items_to_process.append({"id": row['email_id'], "type": "Email", "text": f"Subject: {row['subject']}, Body: {row['body']}"})
        
        items_to_process = items_to_process[:process_limit]
        
        progress_bar = st.progress(0)
        processed_data = []
        processing_logs = []
        
        for i, item in enumerate(items_to_process):
            with st.spinner(f"Processing {item['id']}..."):
                start_time = datetime.now()
                try:
                    res = process_feedback(item['text'], item['type'])
                    end_time = datetime.now()
                    
                    res_str = str(res.raw if hasattr(res, 'raw') else res)
                    
                    import re
                    json_match = re.search(r'\{.*\}', res_str, re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group(0))
                    else:
                        parsed = json.loads(res_str)
                    
                    parsed['source_id'] = item['id']
                    parsed['source_type'] = item['type']
                    processed_data.append(parsed)
                    processing_logs.append({"source_id": item['id'], "status": "Success", "processing_time_sec": (end_time - start_time).seconds})
                except Exception as e:
                    end_time = datetime.now()
                    st.error(f"Failed to process ticket for {item['id']}: {e}")
                    processed_data.append({"source_id": item['id'], "category": "Error", "raw_output": str(e)})
                    processing_logs.append({"source_id": item['id'], "status": f"Error: {e}", "processing_time_sec": (end_time - start_time).seconds})
            
            progress_bar.progress((i + 1) / len(items_to_process))
            
        st.success("Processing Complete!")
        
        st.write("### Generated Tickets")
        res_df = pd.DataFrame(processed_data)
        st.dataframe(res_df)
        
        res_df.to_csv("generated_tickets.csv", index=False)
        st.write("Saved tickets to `generated_tickets.csv`.")
        
        logs_df = pd.DataFrame(processing_logs)
        logs_df.to_csv("processing_log.csv", index=False)
        
        success_count = len([p for p in processing_logs if p['status'] == 'Success'])
        metrics_df = pd.DataFrame({
            "Total_Processed": [len(processed_data)],
            "Success_Count": [success_count],
            "Success_Rate": [success_count / len(processed_data) if processed_data else 0],
            "Average_Time_Sec": [logs_df['processing_time_sec'].mean() if not logs_df.empty else 0]
        })
        metrics_df.to_csv("metrics.csv", index=False)
        
        st.write("### Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Processed", len(processed_data))
        col2.metric("Success Rate", f"{success_count / len(processed_data) * 100:.1f}%" if processed_data else "0%")
        col3.metric("Avg Processing Time", f"{logs_df['processing_time_sec'].mean():.1f}s" if not logs_df.empty else "0s")

st.write("### Manual Override / Generated Tickets View")
if os.path.exists("generated_tickets.csv"):
    df_tickets = pd.read_csv("generated_tickets.csv")
    edited_df = st.data_editor(df_tickets)
    if st.button("Save Changes"):
        edited_df.to_csv("generated_tickets.csv", index=False)
        st.success("Changes saved successfully.")
else:
    st.info("No tickets generated yet. Run the processing to see them here.")
