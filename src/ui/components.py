import streamlit as st
from typing import List

def create_metric_card(title: str, value: str, icon: str, color: str):
    st.markdown(f"""
        <div style='background-color: #1a1c24; padding: 1.5rem; border-radius: 10px; 
                    box-shadow: 0 4px 6px rgba(0,0,0,0.3); border-left: 5px solid {color}; border: 1px solid #333;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <p style='color: #b0b0b0; margin: 0; font-size: 0.9rem;'>{title}</p>
                    <h3 style='color: #ffffff; margin: 0.5rem 0 0 0;'>{value}</h3>
                </div>
                <div style='font-size: 2rem;'>{icon}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def create_camera_card(placeholder, title: str):
    """Helper - in this app we use direct st.image, but this could wrap detailed views"""
    pass

def create_log_viewer(logs: List):
    """Display logs in a text area. Handles both string and dict log entries."""
    formatted_logs = []
    for log in logs:
        if isinstance(log, dict):
            # Format dictionary log entry
            timestamp = log.get('timestamp', 'N/A')
            level = log.get('level', 'INFO')
            message = log.get('message', '')
            formatted_logs.append(f"[{timestamp}] [{level}] {message}")
        else:
            # Handle legacy string format
            formatted_logs.append(str(log))
    
    log_text = "\n".join(formatted_logs)
    st.text_area("Console Output", log_text, height=300, 
                 help="Real-time system event logs")

