import os
import json
import re
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq

def process_feedback_with_crew(feedback_text, api_key):
    """
    Yeh function Streamlit UI se raw feedback aur API key leta hai, 
    aur 6-Agent CrewAI pipeline ke through process karke JSON tickets return karta hai.
    """
    
    # ==========================================
    # 1. LLM INITIALIZATION
    # ==========================================
    # Streamlit UI se aayi API key ka use kar rahe hain taaki app crash na ho
    llm = ChatGroq(
        temperature=0,
        model_name="llama-3.1-8b-instant",
        groq_api_key=api_key
    )

    # ==========================================
    # 2. AGENTS DEFINITION (Total 6 Agents)
    # ==========================================
    
    # Agent 1: CSV Reader Agent
    csv_reader_agent = Agent(
        role='CSV Reader Agent',
        goal='Feedback data ko CSV files se read aur parse karna',
        backstory='Aap data processing expert hain jo structured data ko analyze karne ke liye prepare karte hain.',
        allow_delegation=False,
        llm=llm
    )

    # Agent 2: Feedback Classifier Agent
    classifier_agent = Agent(
        role='Feedback Classifier Agent',
        goal='Feedback ko Bug, Feature Request, Praise, Complaint, ya Spam mein categorize karna',
        backstory='Aap ek experienced product analyst hain jo user sentiments aur categories ko pehchante hain.',
        allow_delegation=False,
        llm=llm
    )

    # Agent 3: Bug Analysis Agent
    bug_analyst_agent = Agent(
        role='Bug Analysis Agent',
        goal='Bugs ke technical details jaise reproduction steps, platform info, aur severity extract karna',
        backstory='Aap ek software QA engineer hain jo system stability aur crash reports par focus karte hain.',
        allow_delegation=False,
        llm=llm
    )

    # Agent 4: Feature Extractor Agent
    feature_extractor_agent = Agent(
        role='Feature Extractor Agent',
        goal='New feature requests identify karna aur unka user impact estimate karna',
        backstory='Aap ek product manager hain jo user needs ko roadmap features mein convert karte hain.',
        allow_delegation=False,
        llm=llm
    )

    # Agent 5: Ticket Creator Agent
    ticket_creator_agent = Agent(
        role='Ticket Creator Agent',
        goal='Structured engineering tickets generate karna aur outputs ko manage karna',
        backstory='Aap ek technical lead hain jo developers ke liye actionable tasks create karte hain.',
        allow_delegation=True, # Needs to collaborate with QA
        llm=llm
    )

    # Agent 6: Quality Critic Agent
    quality_critic_agent = Agent(
        role='Quality Critic Agent',
        goal='Generated tickets ki completeness aur accuracy review karna',
        backstory='Aap ek senior auditor hain jo ensure karte hain ki har ticket strict quality standards meet kare.',
        allow_delegation=False,
        llm=llm
    )

    # ==========================================
    # 3. TASKS DEFINITION (Total 6 Tasks)
    # ==========================================
    
    # Task 1: Reading and Organizing
    read_task = Task(
        description=(
            "Read and organize the following raw feedback data for downstream analysis:\n\n"
            f"{feedback_text}"
        ),
        expected_output="A cleanly formatted text summary of all feedback items with their original IDs.",
        agent=csv_reader_agent
    )

    # Task 2: Classification
    classify_task = Task(
        description=(
            "Review the organized feedback from the previous task. "
            "Categorize each item strictly into one of these: Bug, Feature Request, Praise, Complaint, or Spam."
        ),
        expected_output="A list of categorized feedback items linked to their IDs.",
        agent=classifier_agent
    )

    # Task 3: Bug Extraction
    bug_task = Task(
        description=(
            "Review the categorized items. For every item categorized as 'Bug', "
            "extract the severity (Critical/High/Medium/Low), platform (iOS/Android/Web), and steps to reproduce."
        ),
        expected_output="Detailed technical specifications exclusively for items marked as Bugs.",
        agent=bug_analyst_agent
    )

    # Task 4: Feature Impact Extraction
    feature_task = Task(
        description=(
            "Review the categorized items. For every item categorized as 'Feature Request', "
            "estimate the user impact (High/Medium/Low) and general demand level."
        ),
        expected_output="Impact and demand estimations exclusively for items marked as Feature Requests.",
        agent=feature_extractor_agent
    )

    # Task 5: Ticket Generation
    ticket_task = Task(
        description=(
            "Using the categorization, bug details, and feature impact data from previous tasks, "
            "create structured engineering tickets. Each ticket MUST have a title starting with a tag like [BUG] or [FEATURE], "
            "a description, priority, component, and effort estimation."
        ),
        expected_output="A detailed list of fully drafted engineering tickets.",
        agent=ticket_creator_agent
    )

    # Task 6: Final QA and JSON Output Setup
    qa_task = Task(
        description=(
            "Review all generated tickets from the Ticket Creator for completeness. "
            "Assign a quality_score (1-10) and qa_approved (true/false) boolean to each ticket. "
            "IMPORTANT INSTRUCTION: Your final output MUST be a valid JSON array of objects ONLY. Do not use markdown blocks, do not add introductory text. "
            "Schema: [{\"ticket_id\": \"TKT-...\", \"source_id\": \"...\", \"category\": \"...\", \"priority\": \"...\", \"title\": \"...\", \"description\": \"...\", \"quality_score\": 8, \"qa_approved\": true}]"
        ),
        expected_output="A strict, pure JSON array containing the final approved tickets, ready for parsing.",
        agent=quality_critic_agent
    )

    # ==========================================
    # 4. CREW ASSEMBLY & EXECUTION
    # ==========================================
    feedback_crew = Crew(
        agents=[
            csv_reader_agent, 
            classifier_agent, 
            bug_analyst_agent, 
            feature_extractor_agent, 
            ticket_creator_agent, 
            quality_critic_agent
        ],
        tasks=[
            read_task, 
            classify_task, 
            bug_task, 
            feature_task, 
            ticket_task, 
            qa_task
        ],
        process=Process.sequential, # Tasks sequentially ek ke baad ek challenge
        verbose=True
    )

    print("🚀 CrewAI pipeline is kicking off...")
    result = feedback_crew.kickoff()
    
# ==========================================
    # 5. JSON PARSING & RETURN (Bulletproof Version)
    # ==========================================
    import re # File ke shuru mein 'import re' zaroor check kar lein
    
    try:
        # CrewAI ke alag-alag versions ko handle karne ke liye safe check
        output_str = result.raw if hasattr(result, 'raw') else str(result)
        
        # LLM se aane wale extra markdown tags ya text ko regex se clean karna
        match = re.search(r'\[.*\]', output_str, re.DOTALL)
        if match:
            cleaned_result = match.group(0)
        else:
            cleaned_result = output_str.strip().strip('```json').strip('```')
            
        tickets_json = json.loads(cleaned_result)
        return tickets_json
        
    except Exception as e:
        print(f"Error parsing JSON from CrewAI: {e}")
        output_to_print = result.raw if hasattr(result, 'raw') else str(result)
        print("Raw Output was:", output_to_print)
        return []