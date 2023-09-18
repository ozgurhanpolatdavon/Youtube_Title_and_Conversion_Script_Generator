import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from os import environ
from dotenv import load_dotenv

load_dotenv()

apikey = environ['OPENAI_API_KEY']
print(apikey)

# App framework
st.title("🦜️🔗 Youtube GPT Creator")
prompt = st.text_input("Plug in your prompt here")

# Llms
llm = OpenAI(
    temperature=0.9,
    model_name="gpt-3.5-turbo",
    openai_api_key=apikey,
)

title_template = PromptTemplate(
    input_variables=['topic'],
    template="write me a youtube video title about {topic}"
)

script_template = PromptTemplate(
    input_variables=['title', "wikipedia_research"],
    template="write me a youtube video script based on this title :  {title} while reveraging this wikipedia research : {wikipedia_research}"
)

title_memory = ConversationBufferMemory(input_key='topic', memory_key="chat history")
script_memory = ConversationBufferMemory(input_key='title', memory_key="chat history")

title_chain = LLMChain(llm=llm, prompt=title_template, output_key="title", verbose=True, memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, output_key="script", verbose=True, memory=script_memory)
#sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'],
#                                   output_variables=["title", "script"], verbose=True)


wiki = WikipediaAPIWrapper()

if prompt:
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
