import os
import logging
import google.cloud.logging
from dotenv import load_dotenv

from google.adk.agents.llm_agent import Agent
from google.adk.agents import SequentialAgent
from google.adk.tools.langchain_tool import LangchainTool
from google.adk.tools.tool_context import ToolContext

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain.utilities import PubMedAPIWrapper

import google.auth
import google.auth.transport.requests
import google.oauth2.id_token

load_dotenv()

model_name = os.getenv("MODEL")

# Greet user and save their prompt

def add_prompt_to_state(
    tool_context: ToolContext, prompt: str
) -> dict[str, str]:
    """Saves the user's initial prompt to the state."""
    tool_context.state["PROMPT"] = prompt
    logging.info(f"[State updated] Added to PROMPT: {prompt}")
    return {"status": "success"}

# Configuring the PubMed Tool
wrapper = PubMedAPIWrapper(top_k_results=2)
pubmed_tool = LangchainTool(tool=PubmedQueryRun(api_wrapper=wrapper))

# Configuring the Wikipedia Tool
wikipedia_tool = LangchainTool(
    tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
)

# 1. Researcher Agent
comprehensive_researcher = Agent(
    name="comprehensive_researcher",
    model="gemini-2.5-flash",
    description="The primary researcher that can access external knowledge from Pubmed.",
    instruction="""
    You are a helpful research assistant. Your goal is to answer the user's PROPMT.
    You have two tools:
    1. `wikipedia_tool` : Wikipedia can be used to get general information about any disease of medical context.
    2. `pubmed_tool` : PubMed can be used to get deeper research related to any medical information.

    First, analyze the user's PROMPT.
    - If the prompt can be answered by `wikipedia_tool`, use that tool only.
    - If the prompt need deeper medical research, you MUST use `pubmed_tool` to gather necessary information.
    - Synthesize the results from the tool(s) you use into preliminary data outputs.

    IMPORTANT: Please make sure to get information only ONCE! No iteration!
    
    PROMPT:
    { PROMPT }
    """,
    tools=[
        wikipedia_tool,
        pubmed_tool
    ],
    output_key="research_data" # A key to store the combined findings
)

# 2. Response Formatter Agent
response_formatter = Agent(
    name="response_formatter",
    model=model_name,
    description="Synthesizes all information into a friendly, readable response.",
    instruction="""
    You are the friendly voice of the Medical Buddy Agent. Your task is to take the
    RESEARCH_DATA and present it to the user in a complete, understandable, and helpful answer.

    - First, take medical facts from the research.
    - If some information is missing, just present the information you have.
    - Make sure the answer is easy to understand for non-medical user.
    - Be friendly and engaging.
    - Give disclaimer that you are not a healthcare professional. User need to see the doctor if needed.

    RESEARCH_DATA:
    { research_data }
    """
)

tour_guide_workflow = SequentialAgent(
    name="tour_guide_workflow",
    description="The main workflow for handling a user's request about an medical questions.",
    sub_agents=[
        comprehensive_researcher, # Step 1: Gather all data
        response_formatter,       # Step 2: Format the final response
    ]
)

root_agent = Agent(
    model=model_name,
    name="greeter",
    description="The main entry point for the medical buddy.",
    instruction="""
    - Let the user know you will help them answer any medical question.
    - When the user responds, use the 'add_prompt_to_state' tool to save their response.
    Only transfer to 'tour_guide_workflow' if the user has provided a medical question.
    """,
    tools=[add_prompt_to_state],
    sub_agents=[tour_guide_workflow]
)
