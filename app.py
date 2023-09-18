import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from os import environ
from dotenv import load_dotenv


load_dotenv()

apikey = environ['OPENAI_API_KEY']

# App framework
st.title("🦜️🔗 Youtube GPT Creator")
prompt = st.text_input("Plug in your prompt here")

# Llms
llm = OpenAI(
    temperature=0.9,
    model_name="text-davinci-003",
    openai_api_key=apikey,
)

script_template = PromptTemplate(
    input_variables=['topic'],
    template="write me a youtube video title about {topic} and write me a youtube video script based on the title while reveraging the wikipedia research"
)

agent_memory = ConversationBufferMemory(memory_key="agent memory history")

#sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'],
#                                   output_variables=["title", "script"], verbose=True)

wiki = WikipediaAPIWrapper()

#Custom Agent
tools = [
    Tool(
      name="wikipedia search",
      func=wiki.run,
      description="useful for when you need to get some information about a topic to generate something"
    ),

]
agent_chain = initialize_agent(tools,llm,agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,verbose=True,memory=agent_memory)

if prompt:

    res = agent_chain.run({"input": script_template.format(topic=prompt),"chat_history": []})
    print(res)

    """ 
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)


    #response = sequential_chain({"topic": prompt})3
    st.write(title)
    st.write(script)

    with st.expander("Message History"):
        st.info(title_memory.buffer)
    with st.expander("Script History"):
        st.info(script_memory.buffer)
    with st.expander("Wikipedia History"):
        st.info(wiki_research)
"""