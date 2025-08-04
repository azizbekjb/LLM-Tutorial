import streamlit as st
from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

load_dotenv()
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = st.text_input("Google Gemini uchun API kalitni kiriting", type="password")
try:
    model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    st.success("Model bilan aloqa bog'landi")
except Exception as ex:
    st.error("Xatolik!!!")
    st.error(ex)
    

def generate_resturant_name_and_items(cuisine):
    
    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template="I want to opan a resturant for {cuisine} food. Suggest a fancy name for this. Only one name"
    )

    name_chain = LLMChain(llm=model, prompt=prompt_template_name, output_key="resturant_name")

    prompt_template_items = PromptTemplate(
        input_variables=['resturant_name'],
        template="Suggest some menu items for {resturant_name}. Return it as a comma separated string"
    )
    food_items_chain = LLMChain(llm=model, prompt=prompt_template_items, output_key="menu_items")
    chain = SequentialChain(
        chains = [name_chain, food_items_chain],
        input_variables = ['cuisine'],
        output_variables = ['resturant_name', 'menu_items']
    )
    response = chain({'cuisine' : cuisine})  
    return response

st.title("Resturant Name Generator")
cuisine = st.sidebar.selectbox("Pick a Cuisine", options=("Uzbek", "Indian", "Italian", "Mexican", "American"))

if cuisine:
    try:
        response = generate_resturant_name_and_items(cuisine=cuisine)
        st.header(response["resturant_name"].strip())
        menu_items = response['menu_items'].strip().split(",")
        st.write("**Menu Items**")
        for item in menu_items:
            st.write("-", item)
    except:
        st.error("Model bilan aloqa mavjud emas. Iltimos API kalitni kiriting!")
