import streamlit as st
import os
import json
from typing import Dict, List
from langchain_openai import ChatOpenAI
import httpx
import warnings
warnings.filterwarnings("ignore")

# Configure SSL
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

@st.cache_resource
def initialize_llm(api_key: str):
    client = httpx.Client(verify=False)
    llm = ChatOpenAI(
        base_url="https://genailab.tcs.in",
        model="azure_ai/genailab-maas-DeepSeek-V3-0324",
        api_key=api_key,
        http_client=client,
        temperature=0.3,
        max_tokens=2000
    )
    return llm

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "repair_jobs" not in st.session_state:
        st.session_state.repair_jobs = []

def chat_with_llm(llm, message):
    prompt = f"""
    You are an expert repair technician AI assistant specialized in creating job cards from technical manuals.
    
    User Question: {message}
    
    Provide detailed, technical advice for repair and maintenance tasks.
    If asked about creating job cards, provide structured examples.
    Include safety considerations, tools needed, and step-by-step procedures.
    """
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"

def generate_sample_job_card(llm, equipment_type):
    prompt = f"""
    Create a detailed repair job card for: {equipment_type}
    
    Include:
    1. JOB CARD HEADER (Job ID, Date, Technician, Priority)
    2. EQUIPMENT DETAILS
    3. PROBLEM DESCRIPTION
    4. REQUIRED PARTS
    5. REQUIRED TOOLS
    6. STEP-BY-STEP PROCEDURE
    7. SAFETY PRECAUTIONS
    8. QUALITY CHECKS
    9. NOTES
    
    Make it realistic and detailed.
    """
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    st.set_page_config(
        page_title="Repair Job AI Assistant",
        page_icon="🔧",
        layout="wide"
    )
    
    init_session_state()
    
    st.title("🔧 Repair Job AI Assistant")
    st.markdown("Create repair job cards using AI - No document upload required")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("API Key", type="password", value=st.session_state.api_key)
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            if api_key:
                st.success("API key updated")
        
        st.header("🛠️ Quick Templates")
        templates = [
            "Automotive Brake System Repair",
            "HVAC Maintenance",
            "Industrial Pump Repair",
            "Electrical Panel Maintenance",
            "Hydraulic System Service"
        ]
        
        for template in templates:
            if st.button(f"📋 {template}"):
                if st.session_state.api_key:
                    llm = initialize_llm(st.session_state.api_key)
                    with st.spinner(f"Creating {template} job card..."):
                        job_card = generate_sample_job_card(llm, template)
                        st.session_state.repair_jobs.append({
                            "title": template,
                            "job_card": job_card,
                            "timestamp": "Now"
                        })
                        st.success("Job card created!")
                else:
                    st.error("Please enter API key")
    
    # Main area
    tab1, tab2 = st.tabs(["💬 Chat", "📋 Job Cards"])
    
    with tab1:
        st.header("Chat with Repair Expert AI")
        
        if not st.session_state.api_key:
            st.warning("Enter your API key in the sidebar to start")
        else:
            llm = initialize_llm(st.session_state.api_key)
            
            # Show chat history
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask about repairs, job cards, or technical procedures..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Get response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = chat_with_llm(llm, prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Clear chat button
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.rerun()
    
    with tab2:
        st.header("Generated Job Cards")
        
        if not st.session_state.repair_jobs:
            st.info("No job cards yet. Use templates in sidebar or ask AI to create one.")
        else:
            for i, job in enumerate(st.session_state.repair_jobs):
                with st.expander(f"🛠️ {job['title']}", expanded=(i==0)):
                    st.markdown(job['job_card'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "📥 Download",
                            job['job_card'],
                            file_name=f"{job['title'].replace(' ', '_')}_job_card.md",
                            mime="text/markdown",
                            key=f"dl_{i}"
                        )
                    with col2:
                        if st.button("🗑️ Delete", key=f"del_{i}"):
                            st.session_state.repair_jobs.pop(i)
                            st.rerun()
    
    # Footer
    st.markdown("---")
    st.caption("Repair Job AI Assistant | Create job cards without document upload")

if __name__ == "__main__":
    main()