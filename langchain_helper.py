from langchain_openai import OpenAI  # Import from langchain-openai
from langchain.prompts import PromptTemplate
from langchain.agents import load_tools, initialize_agent, AgentType
from dotenv import load_dotenv

load_dotenv()

def generate_pet_name(animal_type, pet_color):
    # Use the updated OpenAI class
    llm = OpenAI(temperature=0.7)

    prompt_template_name = PromptTemplate(
        input_variables=['animal_type', 'pet_color'],
        template="I have a pet {animal_type} and I want a cool name for it, it is {pet_color} in color. Suggest me five cool names for my pet"
    )

    # Chain the prompt template and LLM using the pipe (|) operator
    name_chain = prompt_template_name | llm  # No need for LLMChain, use pipe (|) operator

    # Use invoke() instead of __call__
    response = name_chain.invoke({'animal_type': animal_type, 'pet_color': pet_color})

    return response

def langchain_agent():
    llm = OpenAI(temperature=0.5)  # Use OpenAI for the agent

    # Load the tools, specifically the 'wikipedia' tool
    tools = load_tools(["wikipedia"], llm=llm)  # Make sure you pass the correct tool name
    
    # Initialize the agent with the tools and the LLM
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    # Run the agent with a question to get the average age of a dog
    result = agent.run("What is the average lifespan of a dog?")
    
    # Print the result
    print(result)



if __name__ == "__main__":
    langchain_agent()
    #print(generate_pet_name("cat", "red"))
