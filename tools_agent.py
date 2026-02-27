import requests
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

tasks = []

# =========================
# TOOLS
# =========================

@tool
def get_weather(city: str) -> str:
    """
    Get current weather for a given city.
    """
    # Example using free Open-Meteo API
    try:
        geo = requests.get(
            f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
        ).json()

        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]

        weather = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        ).json()

        temp = weather["current_weather"]["temperature"]
        wind = weather["current_weather"]["windspeed"]

        return f"The current temperature in {city} is {temp}°C with wind speed {wind} km/h."
    except:
        return "Sorry, I couldn't fetch weather data."


@tool
def add_task(task: str) -> str:
    """
    Add a task to the to-do list.
    """
    tasks.append(task)
    return f"Task added: {task}"


@tool
def list_tasks() -> str:
    """
    List all tasks in the to-do list.
    """
    if not tasks:
        return "Your to-do list is empty."
    return "Your tasks:\n" + "\n".join([f"{i+1}. {t}" for i, t in enumerate(tasks)])


@tool
def calculate(expression: str) -> str:
    """
    Calculate a mathematical expression.
    """
    try:
        result = eval(expression)
        return f"result: {result}"
    except:
        return "Sorry, I couldn't calculate that expression."
    

@tool
def current_time() -> str:
    """
    Get the current time.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"The current time is {now}."



llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a funny personal assistant that helps users but also jokes around."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

memory = ConversationBufferMemory(memory_key="assistant", return_messages=True)

tools = [get_weather, add_task, list_tasks, calculate, current_time]

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# =========================
# RUN LOOP
# =========================

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        response = agent_executor.invoke({"input": user_input})
        print(f"Assistant: {response}")


"""
Sample interactions:
What's the weather in London?
Add buy groceries to my tasks
List my tasks
What is 45 * 12?
What time is it?
"""
