from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain.agents import load_tools, initialize_agent

load_dotenv()  # ensure OPENAI_API_KEY is loaded

llm = OpenAI(temperature=0)

tools = load_tools(["wikipedia", "llm-math"], llm=llm)

agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=False
)

response = agent.run("What is GDP of US and Canada separately? What is their GDP combined?")
print(response)
