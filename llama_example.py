import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Define a simple function to be used by the agent
def multiply(a: int, b: int) -> int:
    """Useful for multiplying two numbers."""
    print(f"Tool called! a={a}, b={b}")
    return a * b

agent = FunctionAgent(
    llm=OpenAI(temperature=0),
    tools=[multiply],
    system_prompt="You are a helpful assistant that can multiply two numbers."
)

async def main():
    respone = await agent.run("What is the product of 1387 and 2946?")
    print(str(respone))

if __name__ == "__main__":
    asyncio.run(main())
