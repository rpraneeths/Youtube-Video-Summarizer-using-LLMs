import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.llms import HuggingFaceHub

# Load environment variables
load_dotenv()

# Setup the HuggingFace LLM
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",  # A good open-source LLM for text generation
    model_kwargs={"temperature": 0.5, "max_length": 256}
)

def refine_prompt(base_prompt: str) -> str:
    """
    Uses an agent to optimize a base prompt for better summarization.
    """
    agent = initialize_agent(
        tools=[],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    improvement_instruction = f"""
    You are a prompt optimization expert.
    Improve the following prompt for summarizing content:
    - Make it structured (Title, Key Points, Summary)
    - Limit summary to 150 words
    - Keep clarity and context
    - Return only the improved prompt

    Base Prompt:
    {base_prompt}
    """

    return agent.run(improvement_instruction)


if __name__ == "__main__":
    base_prompt = "Summarize the following text about climate change."
    optimized_prompt = refine_prompt(base_prompt)
    print("\nOptimized Prompt:\n", optimized_prompt)
