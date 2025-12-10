import streamlit as st

def apply_custom_css():
    st.markdown("""

        <style>
        /* ----------------------------------------------------
           MAIN THEME: Scaler_app Inspired
           ---------------------------------------------------- */
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            background-attachment: fixed;
        }
        
        .stApp {
            background-color: #0e1117; /* Fallback */
        }

        /* Glassmorphic Container */
        .block-container {
            background: rgba(17, 24, 39, 0.95);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-top: 2rem;
        }

        /* Gradient Headings */
        h1 {
            background: linear-gradient(135deg, #a78bfa 0%, #f472b6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem !important;
            font-weight: 800 !important;
            text-align: center;
            margin-bottom: 1rem;
            animation: fadeInDown 1s ease-in-out;
        }
        h2 { 
            color: #f3f4f6 !important; 
            border-bottom: 3px solid #764ba2; 
            padding-bottom: 0.5rem; 
            margin-top: 2rem; 
            font-weight: 700 !important; 
        }
        h3 { 
            color: #e5e7eb !important; 
            margin-top: 1.5rem; 
            font-weight: 600 !important; 
        }
        p, li, span, div { color: #d1d5db; }

        /* Metrics styling */
        [data-testid="stMetricValue"] {
            background: linear-gradient(135deg, #a78bfa 0%, #f472b6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2rem !important;
        }
        [data-testid="stMetricLabel"] { color: #9ca3af !important; }

        /* Custom Tabs */
        .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: transparent; }
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(102, 126, 234, 0.1); 
            color: #667eea; 
            border-radius: 8px; 
            padding: 10px 20px; 
            font-weight: 600; 
            transition: all 0.3s ease; 
            border: 1px solid rgba(102, 126, 234, 0.2);
        }
        .stTabs [data-baseweb="tab"]:hover { 
            background-color: rgba(102, 126, 234, 0.2); 
            transform: translateY(-2px); 
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; 
            color: white !important; 
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
            border: none;
        }

        /* Sidebar */
        [data-testid="stSidebar"] { 
            background: linear-gradient(180deg, #1a1c24 0%, #0e1117 100%); 
            color: white; 
            border-right: 1px solid rgba(255,255,255,0.1);
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            border-radius: 10px; 
            padding: 0.75rem 2rem; 
            font-weight: 600; 
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            border: none;
        }
        .stButton > button:hover { 
            transform: translateY(-2px); 
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6); 
            color: white;
        }

        /* Animations */
        @keyframes fadeInDown { 
            from { opacity: 0; transform: translateY(-20px); } 
            to { opacity: 1; transform: translateY(0); } 
        }

        /* ----------------------------------------------------
           PROJECT SPECIFIC COMPONENTS
           ---------------------------------------------------- */
        
        /* Surveillance Frame */
        .surveillance-frame {
            border: 2px solid #333;
            border-radius: 4px;
            position: relative;
            box-shadow: 0 0 15px rgba(0, 255, 0, 0.1);
            overflow: hidden;
            background: #000;
        }
        .surveillance-frame::before {
            content: "REC ●";
            position: absolute;
            top: 10px; right: 15px;
            color: #ff4b4b;
            font-weight: bold; font-size: 12px;
            animation: blink 1s infinite;
            z-index: 10;
        }
        .surveillance-frame::after {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            border: 1px solid rgba(100, 255, 100, 0.3);
            pointer-events: none;
        }
        @keyframes blink { 50% { opacity: 0; } }

        /* Tech Stack Badges */
        .stack-badge {
            background: linear-gradient(135deg, #2b313e 0%, #1a1c24 100%);
            border: 1px solid #4a4d5a;
            padding: 10px 15px;
            border-radius: 20px;
            margin: 5px;
            display: inline-block;
            font-size: 0.9rem;
            color: #b0b0b0;
            transition: all 0.2s;
        }
        .stack-badge:hover {
            border-color: #667eea;
            color: white;
            transform: scale(1.05);
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }

        /* Timeline Steps (How It Works) */
        .step-container {
            background: rgba(255,255,255,0.03);
            border-left: 3px solid #667eea;
            padding: 15px 20px;
            margin-bottom: 15px;
            border-radius: 0 10px 10px 0;
        }
        .step-number {
             font-size: 1.5rem;
             font-weight: bold;
             color: #667eea;
             margin-bottom: 5px;
        }

        /* Command Center Box */
        .command-box {
            background: rgba(20, 22, 31, 0.8);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 15px;
            height: 100%;
        }
        .command-header {
            border-bottom: 1px solid rgba(255,255,255,0.1);
            padding-bottom: 10px;
            margin-bottom: 15px;
            color: #a78bfa;
            font-weight: bold;
            font-size: 1.1rem;
        }
        </style>
    """, unsafe_allow_html=True)

def get_gradient_background(color1: str, color2: str) -> str:
    return f"background: linear-gradient(135deg, {color1} 0%, {color2} 100%);"
