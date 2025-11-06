
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0.7)

prompt_template_name = PromptTemplate.from_template("Suggest a fancy name for {cuisine} restaurant?")

name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")

prompt_template_items = PromptTemplate.from_template(
    "Suggest some menu items for {restaurant_name}. Return it as a comma separated list"
)

food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")

chain = SequentialChain(chains=[name_chain, food_items_chain],
                        input_variables=['cuisine'],
                        output_variables=['restaurant_name', 'menu_items'])

cuisine = input("Enter the type of cuisine: ")
response = chain.invoke({"cuisine": cuisine})
print(f"Restaurant: {response['restaurant_name'].strip()}")
print("Menu Items:")
print(response['menu_items'].strip())
