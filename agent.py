import streamlit as st
import google.generativeai as genai
import time
import json
import os
from datetime import datetime
import pandas as pd
import uuid
import plotly.express as px

# Configuration and setup
st.set_page_config(
    page_title="Agentic AI Task Automation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'task_history' not in st.session_state:
    st.session_state.task_history = []
if 'current_task_id' not in st.session_state:
    st.session_state.current_task_id = None
if 'agent_messages' not in st.session_state:
    st.session_state.agent_messages = []
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'planner_model': "gemini-1.5-flash",
        'executive_model': "gemini-1.5-flash",
        'researcher_model': "gemini-1.5-flash",
        'critic_model': "gemini-1.5-flash",
        'max_steps': 10,
        'temperature': 0.7,
        'verbose': True
    }

# Function to save task history to file
def save_history():
    with open('task_history.json', 'w') as f:
        json.dump(st.session_state.task_history, f)

# Function to load task history from file
def load_history():
    if os.path.exists('task_history.json'):
        with open('task_history.json', 'r') as f:
            st.session_state.task_history = json.load(f)

# Load history at startup
try:
    load_history()
except:
    st.session_state.task_history = []

# Agent functions
class Agent:
    def __init__(self, role, model_name, temperature=0.7):
        self.role = role
        self.model_name = model_name
        self.temperature = temperature

    def generate_response(self, prompt):
        try:
            genai.configure(api_key=st.session_state.api_key)
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={"temperature": self.temperature}
            )
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"{self.role} Agent Error: {str(e)}")
            return f"Error in {self.role} agent: {str(e)}"

class PlannerAgent(Agent):
    def plan_task(self, task):
        prompt = f"""As a Task Planning Agent, break down this task into detailed, executable steps. 
        Format the steps as a numbered list with clear instructions.
        
        Task: {task}
        
        Provide a comprehensive plan that covers all aspects of the task."""
        
        steps = self.generate_response(prompt)
        log_agent_message("Planner", f"I've broken down the task into these steps:\n\n{steps}")
        return steps

class ResearcherAgent(Agent):
    def gather_information(self, task, steps):
        prompt = f"""As a Research Agent, gather relevant information to help complete this task.
        
        Task: {task}
        Steps: {steps}
        
        Provide key information, facts, or data that would be helpful for executing this task."""
        
        research = self.generate_response(prompt)
        log_agent_message("Researcher", f"I've gathered this relevant information:\n\n{research}")
        return research

class ExecutiveAgent(Agent):
    def execute_task(self, task, steps, research):
        prompt = f"""As an Executive Agent, execute the given steps based on the task description and research provided.
        
        Task: {task}
        Steps: {steps}
        Research: {research}
        
        Provide the complete solution with detailed execution of each step."""
        
        execution = self.generate_response(prompt)
        log_agent_message("Executive", f"I've executed the task. Here's the result:\n\n{execution}")
        return execution

class CriticAgent(Agent):
    def evaluate_solution(self, task, steps, research, execution):
        prompt = f"""As a Critic Agent, evaluate the solution provided by the Executive Agent.
        
        Task: {task}
        Steps: {steps}
        Research: {research}
        Execution: {execution}
        
        Provide constructive feedback, identify any issues or areas for improvement, and suggest refinements."""
        
        critique = self.generate_response(prompt)
        log_agent_message("Critic", f"Here's my evaluation of the solution:\n\n{critique}")
        return critique

class RefinerAgent(Agent):
    def refine_solution(self, task, execution, critique):
        prompt = f"""As a Refiner Agent, improve the solution based on the critique provided.
        
        Task: {task}
        Current Solution: {execution}
        Critique: {critique}
        
        Provide an improved and refined solution that addresses the issues identified in the critique."""
        
        refinement = self.generate_response(prompt)
        log_agent_message("Refiner", f"I've refined the solution:\n\n{refinement}")
        return refinement

# Helper functions
def log_agent_message(agent_name, message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.agent_messages.append({
        "agent": agent_name,
        "message": message,
        "timestamp": timestamp
    })

def generate_task_id():
    return str(uuid.uuid4())[:8]

def run_agents(task):
    # Create a new task record
    task_id = generate_task_id()
    st.session_state.current_task_id = task_id
    st.session_state.agent_messages = []
    
    task_record = {
        "id": task_id,
        "task": task,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "steps": "",
        "research": "",
        "execution": "",
        "critique": "",
        "refinement": "",
        "completion_time": 0
    }
    
    start_time = time.time()
    
    # Initialize agents
    planner = PlannerAgent("Planner", st.session_state.settings['planner_model'], st.session_state.settings['temperature'])
    researcher = ResearcherAgent("Researcher", st.session_state.settings['researcher_model'], st.session_state.settings['temperature'])
    executive = ExecutiveAgent("Executive", st.session_state.settings['executive_model'], st.session_state.settings['temperature'])
    critic = CriticAgent("Critic", st.session_state.settings['critic_model'], st.session_state.settings['temperature'])
    refiner = RefinerAgent("Refiner", st.session_state.settings['executive_model'], st.session_state.settings['temperature'])
    
    # Run the agent workflow
    with st.status("Running AI Agents...", expanded=True) as status:
        # Step 1: Planning
        status.update(label="Planning task steps...")
        task_record["steps"] = planner.plan_task(task)
        
        # Step 2: Research
        status.update(label="Gathering relevant information...")
        task_record["research"] = researcher.gather_information(task, task_record["steps"])
        
        # Step 3: Execution
        status.update(label="Executing the task...")
        task_record["execution"] = executive.execute_task(task, task_record["steps"], task_record["research"])
        
        # Step 4: Critique
        status.update(label="Evaluating the solution...")
        task_record["critique"] = critic.evaluate_solution(task, task_record["steps"], task_record["research"], task_record["execution"])
        
        # Step 5: Refinement
        status.update(label="Refining the solution...")
        task_record["refinement"] = refiner.refine_solution(task, task_record["execution"], task_record["critique"])
        
        status.update(label="Task completed!", state="complete")
    
    # Calculate completion time
    task_record["completion_time"] = round(time.time() - start_time, 2)
    
    # Add to history
    st.session_state.task_history.append(task_record)
    save_history()
    
    return task_record

# UI Components
def sidebar_ui():
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # API Key Input
        api_key = st.text_input(
            "Gemini API Key", 
            value="AIzaSyBj1BzzNCg6FOUeic8DTtU3uYNVMaDErQw",
            type="password"
        )
        st.session_state.api_key = api_key
        
        st.divider()
        
        # Model Settings
        st.subheader("Model Settings")
        st.session_state.settings['planner_model'] = st.selectbox(
            "Planner Model",
            ["gemini-1.5-flash", "gemini-1.5-pro"],
            index=0
        )
        st.session_state.settings['executive_model'] = st.selectbox(
            "Executive Model",
            ["gemini-1.5-flash", "gemini-1.5-pro"],
            index=1
        )
        st.session_state.settings['temperature'] = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1
        )
        
        st.divider()
        
        # Task History
        st.subheader("Task History")
        if st.session_state.task_history:
            for i, task in enumerate(reversed(st.session_state.task_history[-5:])):
                if st.button(f"{task['timestamp']}: {task['task'][:20]}...", key=f"history_{i}"):
                    st.session_state.current_task_id = task['id']
        else:
            st.write("No task history yet")
            
        if st.button("Clear History"):
            st.session_state.task_history = []
            save_history()
            st.rerun()

def main_area():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.title("🚀 Advanced Agentic AI Task Automation")
        st.write("Enter a complex task and let the AI agent team break it down, research, execute, and refine the solution.")
        
        task = st.text_area("Enter a Task to Automate:", height=100)
        
        col_btn1, col_btn2 = st.columns([1, 3])
        with col_btn1:
            if st.button("Run AI Agents", type="primary", use_container_width=True):
                if task:
                    task_record = run_agents(task)
                    st.rerun()
                else:
                    st.warning("Please enter a task.")

def display_results():
    if st.session_state.current_task_id is not None:
        # Find the task record
        task_record = None
        for task in st.session_state.task_history:
            if task["id"] == st.session_state.current_task_id:
                task_record = task
                break
        
        if task_record:
            st.header(f"Task: {task_record['task']}")
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Steps", "Research", "Execution", "Critique", "Final Solution"])
            
            with tab1:
                st.subheader("📋 Planned Steps")
                st.write(task_record["steps"])
            
            with tab2:
                st.subheader("🔍 Research")
                st.write(task_record["research"])
            
            with tab3:
                st.subheader("⚙️ Execution")
                st.write(task_record["execution"])
            
            with tab4:
                st.subheader("🧐 Critique")
                st.write(task_record["critique"])
            
            with tab5:
                st.subheader("✨ Final Refined Solution")
                st.write(task_record["refinement"])
            
            # Agent Communication Log
            st.subheader("🤖 Agent Communication Log")
            for message in st.session_state.agent_messages:
                with st.chat_message(message["agent"]):
                    st.write(f"**{message['timestamp']}**")
                    st.write(message["message"])

# Main app layout
def main():
    sidebar_ui()
    main_area()
    
    if st.session_state.task_history:
        st.divider()
        display_results()

if __name__ == "__main__":
    main()