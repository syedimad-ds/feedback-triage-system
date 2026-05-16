import os
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Dict, List
from crew_agents import process_feedback_with_crew

st.set_page_config(
    page_title="Feedback Analysis System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ───────────────────────────────────────────────────────────────
MODEL_NAME = "llama-3.1-8b-instant"

TICKETS_OUTPUT = "generated_tickets.csv"
LOG_OUTPUT = "processing_log.csv"

CATEGORY_COLORS = {
    "Bug": "#FF4B4B",
    "Feature Request": "#4B9BFF",
    "Praise": "#00C853",
    "Complaint": "#FF9800",
    "Spam": "#9E9E9E",
}
PRIORITY_COLORS = {
    "Critical": "#D32F2F",
    "High": "#F57C00",
    "Medium": "#1976D2",
    "Low": "#388E3C",
}
VALID_CATS = {"Bug", "Feature Request", "Praise", "Complaint", "Spam"}

# ── Session state initialization ────────────────────────────────────────
for key, default in {
    "tickets": [],
    "logs": [],
    "pipeline_ran": False,
    "api_key_set": False,
    "groq_client": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Data loading helper ───────────────────────────────────────────────────
def safe_get(row, col, default=""):
    val = row.get(col, default)
    return default if (val is None or (isinstance(val, float) and val != val)) else val

def read_csvs(reviews_df: pd.DataFrame, emails_df: pd.DataFrame) -> List[Dict]:
    items = []
    if reviews_df is not None:
        for _, row in reviews_df.iterrows():
            r = row.to_dict()
            items.append({
                "source_id": str(safe_get(r, "review_id")),
                "source_type": "app_store_review",
                "text": str(safe_get(r, "review_text")),
            })
    if emails_df is not None:
        for _, row in emails_df.iterrows():
            r = row.to_dict()
            items.append({
                "source_id": str(safe_get(r, "email_id")),
                "source_type": "support_email",
                "text": str(safe_get(r, "body")),
            })
    return items

# ── UI helpers ───────────────────────────────────────────────────────────────
def badge(text: str, color: str) -> str:
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:10px;font-size:0.78em;font-weight:600">{text}</span>'

def metric_card(label, value, color="#1976D2"):
    st.markdown(
        f'<div style="background:#1e1e2e;border-left:4px solid {color};padding:12px 16px;border-radius:6px;margin:4px 0">'
        f'<div style="font-size:1.6em;font-weight:700;color:{color}">{value}</div>'
        f'<div style="font-size:0.82em;color:#aaa">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...", value=os.getenv("GROQ_API_KEY", ""))
    if st.button("🔗 Connect", use_container_width=True):
        if api_key:
            try:
                from groq import Groq as _Groq
                client = _Groq(api_key=api_key)
                # Verify key works
                client.chat.completions.create(model=MODEL_NAME, max_tokens=10, messages=[{"role": "user", "content": "hi"}])
                st.session_state.groq_client = client
                st.session_state.api_key_set = True
                st.success("✅ Connected!")
            except Exception as e:
                st.error(f"❌ {e}")
        else:
            st.warning("Paste your API key first.")
    st.markdown("---")
    st.markdown(f"**Model:** `{MODEL_NAME}`")
    st.markdown("**Framework:** `CrewAI`")
    
    if st.session_state.pipeline_ran and st.session_state.tickets:
        st.markdown("---")
        st.markdown("### 📥 Download Outputs")
        df_t = pd.DataFrame(st.session_state.tickets)
        st.download_button("⬇️ generated_tickets.csv", df_t.to_csv(index=False), TICKETS_OUTPUT, "text/csv", use_container_width=True)
        df_l = pd.DataFrame(st.session_state.logs)
        st.download_button("⬇️ processing_log.csv", df_l.to_csv(index=False), LOG_OUTPUT, "text/csv", use_container_width=True)

# ── Main ───────────────────────────────────────────────────────────────────────
st.title("🤖 Intelligent Feedback Analysis System")
st.caption("Multi-agent CrewAI pipeline · Groq API")

tab_run, tab_tickets, tab_accuracy, tab_override = st.tabs(["▶️  Run Pipeline", "🎫 Tickets", "📊 Accuracy", "✏️  Override"])

# ── Run tab ───────────────────────────────────────────────────────────────────
with tab_run:
    col1, col2 = st.columns(2)
    with col1:
        reviews_file = st.file_uploader("📱 app_store_reviews.csv", type="csv", key="rev")
    with col2:
        emails_file = st.file_uploader("📧 support_emails.csv", type="csv", key="em")
    expected_file = st.file_uploader("✅ expected_classifications.csv (optional)", type="csv", key="exp")
    if expected_file:
        st.session_state["expected_df"] = pd.read_csv(expected_file)
    st.markdown("---")
    
    if st.button("▶️  Run Pipeline", type="primary", disabled=not st.session_state.api_key_set, use_container_width=True):
        if not reviews_file or not emails_file:
            st.error("Upload both CSV files first.")
        else:
            reviews_df = pd.read_csv(reviews_file)
            emails_df = pd.read_csv(emails_file)
            
            # Logging UI setup
            log_box = st.empty()
            logs = []
            def ui_log(msg):
                logs.append(msg)
                log_box.code("\n".join(logs))
                
            ui_log(f"📖 Reading CSVs... ({len(reviews_df)} reviews, {len(emails_df)} emails)")
            items = read_csvs(reviews_df, emails_df)
            ui_log(f"✅ {len(items)} items loaded")
            
            ui_log("\n🚀 Starting 6-Agent CrewAI Pipeline...")
            
            # Format data for CrewAI
            feedback_text_block = ""
            for it in items:
                feedback_text_block += f"ID: {it['source_id']} | Type: {it['source_type']} | Text: {it['text'][:300]}\n"
            
            with st.spinner("🤖 CrewAI Agents are collaborating... Please wait."):
                try:
                    # Execute CrewAI backend
                    api_key_for_crew = st.session_state.groq_client.api_key
                    tickets = process_feedback_with_crew(feedback_text_block, api_key_for_crew)
                    
                    if not tickets:
                        st.error("CrewAI returned an empty result. Please check the logs.")
                        st.stop()
                        
                    for t in tickets:
                        ui_log(f"  ✅ TKT for {t.get('source_id', 'Unknown')} created & approved by Quality Critic")
                        
                except Exception as e:
                    st.error(f"Pipeline failed: {e}")
                    st.stop()
            
            ui_log(f"\n✅ Multi-Agent orchestration complete!")
            
            # Build processing log
            proc_logs = []
            for t in tickets:
                proc_logs.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "source_id": t.get("source_id", ""),
                    "stage": "CrewAI Pipeline",
                    "status": "OK",
                    "detail": t.get("title", "")[:60],
                })
                
            # Update session state
            st.session_state.tickets = tickets
            st.session_state.logs = proc_logs
            st.session_state.pipeline_ran = True
            
            st.success(f"✅ Pipeline complete! {len(tickets)} tickets generated.")
            st.rerun()

    if st.session_state.pipeline_ran and st.session_state.tickets:
        tickets = st.session_state.tickets
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Total Tickets", len(tickets), "#1976D2")
        with c2: metric_card("Bugs", sum(1 for t in tickets if t.get("category") == "Bug"), "#FF4B4B")
        with c3: metric_card("Features", sum(1 for t in tickets if t.get("category") == "Feature Request"), "#4B9BFF")
        with c4: metric_card("QA Approved", sum(1 for t in tickets if t.get("qa_approved")), "#00C853")

# ── Tickets tab ────────────────────────────────────────────────────────────────
with tab_tickets:
    if not st.session_state.pipeline_ran or not st.session_state.tickets:
        st.info("Run the pipeline first.")
    else:
        tickets = st.session_state.tickets
        fc1, fc2 = st.columns(2)
        with fc1:
            cat_options = ["All"] + sorted({t.get("category", "Misc") for t in tickets})
            cat_filter = st.selectbox("Category", cat_options)
        with fc2:
            pri_options = ["All"] + ["Critical", "High", "Medium", "Low"]
            pri_filter = st.selectbox("Priority", pri_options)
            
        filtered = [t for t in tickets if (cat_filter == "All" or t.get("category") == cat_filter) and (pri_filter == "All" or t.get("priority") == pri_filter)]
        st.markdown(f"Showing **{len(filtered)}** tickets")
        
        for t in filtered:
            cat_color = CATEGORY_COLORS.get(t.get("category"), "#888")
            pri_color = PRIORITY_COLORS.get(t.get("priority"), "#888")
            with st.expander(f"{t.get('ticket_id', 'TKT')} — {t.get('title', 'No Title')}", expanded=False):
                left, right = st.columns([3, 1])
                with left:
                    st.markdown(badge(t.get("category", ""), cat_color) + "  " + badge(t.get("priority", ""), pri_color) + "  " + badge(f"Q={t.get('quality_score', 0)}", "#555"), unsafe_allow_html=True)
                    st.markdown(f"**Description:** {t.get('description', '')}")
                    st.caption(f"Source: {t.get('source_id', '')} · QA Approved: {t.get('qa_approved', False)}")
                with right:
                    qa_color = "#00C853" if t.get("qa_approved") else "#FF4B4B"
                    st.markdown(badge("QA ✓" if t.get("qa_approved") else "QA ✗", qa_color), unsafe_allow_html=True)

# ── Accuracy tab ──────────────────────────────────────────────────────────────
with tab_accuracy:
    if not st.session_state.pipeline_ran or not st.session_state.tickets:
        st.info("Run the pipeline first.")
    elif "expected_df" not in st.session_state:
        st.info("Upload expected_classifications.csv on the Run tab.")
    else:
        generated_df = pd.DataFrame(st.session_state.tickets)
        expected_df = st.session_state.expected_df
        
        if "source_id" in generated_df.columns:
            merged = generated_df.merge(expected_df[["source_id", "category", "priority"]].rename(columns={"category": "expected_category", "priority": "expected_priority"}), on="source_id", how="inner")
            total = len(merged)
            if total == 0:
                st.warning("No matching source_ids found between generated tickets and expected CSV.")
            else:
                cat_ok = (merged["category"] == merged["expected_category"]).sum()
                pri_ok = (merged["priority"] == merged["expected_priority"]).sum()
                c1, c2, c3 = st.columns(3)
                with c1: metric_card("Items compared", total, "#1976D2")
                with c2: metric_card("Category accuracy", f"{100*cat_ok/total:.1f}%", "#00C853")
                with c3: metric_card("Priority accuracy", f"{100*pri_ok/total:.1f}%", "#FF9800")
                
                st.markdown("### Comparison Table")
                display = merged[["source_id", "category", "expected_category", "priority", "expected_priority"]].copy()
                st.dataframe(display, use_container_width=True)
        else:
             st.warning("CrewAI did not output the 'source_id' field. Accuracy cannot be calculated.")

# ── Override tab ───────────────────────────────────────────────────────────────
with tab_override:
    if not st.session_state.pipeline_ran or not st.session_state.tickets:
        st.info("Run the pipeline first.")
    else:
        st.markdown("Edit a ticket and click Save to override it in session.")
        tickets = st.session_state.tickets
        ticket_ids = [t.get("ticket_id", "Unknown") for t in tickets]
        sel = st.selectbox("Select ticket", ticket_ids)
        ticket = next((t for t in tickets if t.get("ticket_id") == sel), None)
        
        if ticket:
            col1, col2 = st.columns(2)
            with col1:
                new_title = st.text_input("Title", value=ticket.get("title", ""))
                try:
                    pri_idx = ["Critical", "High", "Medium", "Low"].index(ticket.get("priority", "Medium"))
                except ValueError:
                    pri_idx = 2
                new_priority = st.selectbox("Priority", ["Critical", "High", "Medium", "Low"], index=pri_idx)
                
                try:
                    cat_idx = list(VALID_CATS).index(ticket.get("category", "Bug"))
                except ValueError:
                    cat_idx = 0
                new_category = st.selectbox("Category", list(VALID_CATS), index=cat_idx)
                
            with col2:
                new_desc = st.text_area("Description", value=ticket.get("description", ""), height=120)
                
            if st.button("💾 Save Override", type="primary"):
                for t in st.session_state.tickets:
                    if t.get("ticket_id") == sel:
                        t["title"] = new_title
                        t["priority"] = new_priority
                        t["category"] = new_category
                        t["description"] = new_desc
                        break
                st.success(f"✅ {sel} updated!")
                st.rerun()