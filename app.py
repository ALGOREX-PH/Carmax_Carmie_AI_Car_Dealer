import os
import openai
import pandas as pd
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention
warnings.filterwarnings("ignore")


st.set_page_config(page_title="Carmie by Generative Labs", page_icon=":car:", layout="wide")

with st.sidebar :
    st.title("Generative Labs")

    openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
    if not (openai.api_key.startswith('sk-') and len(openai.api_key)==51):
        st.warning('Please enter your credentials!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')
    with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()

    options = option_menu(
        "Dashboard", 
        ["Home", "Model"],
        icons = ['book', 'tools'],
        menu_icon = "book", 
        default_index = 0,
        styles = {
            "icon" : {"color" : "#dec960", "font-size" : "20px"},
            "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
            "nav-link-selected" : {"background-color" : "#262730"}          
        })


if options == "Home" :
   st.title("Introducing Carmie: Carmax's AI Car Dealer")
   st.write("We are thrilled to introduce you to Carmie, Carmax's very own AI Car Dealer. Carmie is here to revolutionize your car-buying experience with cutting-edge technology and unparalleled convenience.")
   st.write("# What is Carmie?")
   st.write("Carmie is an intelligent AI-powered assistant designed to make your car shopping journey smooth, personalized, and enjoyable. Whether you’re looking to buy your first car, upgrade to a newer model, or simply explore options, Carmie is here to assist you every step of the way.")
   st.write("# How Can Carmie Help You?")
   st.write("1. Personalized Recommendations: Carmie analyzes your preferences, budget, and needs to recommend the best vehicles for you.")
   st.write("2. Instant Information: Get detailed information on car models, features, prices, and availability in real-time.")
   st.write("3. Financing Assistance: Carmie provides guidance on financing options and helps you find the best deals.")
   st.write("# Why Choose Carmie?")
   st.write("- 24/7 Availability: Carmie is always ready to help, anytime, anywhere.")
   st.write("- Expert Knowledge: Benefit from Carmie’s extensive knowledge of all car models and features.")
   st.write("- Hassle-Free Experience: Skip the dealership visit and manage everything online with ease.")
   st.write("- Customer-Centric Approach: Carmie focuses on your needs and provides a tailored car-buying journey.")
   st.write("# Join the Future of Car Shopping")
   st.write("With Carmie, Carmax is taking a bold step into the future of car shopping. Embrace the convenience, efficiency, and personalization that only an AI-driven assistant can offer. Discover your perfect car today with Carmie!")

elif options == "Model" :
     dataframed = pd.read_csv('https://raw.githubusercontent.com/ALGOREX-PH/Carmax_Carmie_AI_Car_Dealer/main/Dataset/Carmax.csv')

     Container = """

     # Carmax Dealership Car List
    
     """
     columns = dataframed.columns

     for x in range(0, len(dataframed)) :
         temp = ""
         for y in range(0, len(columns)) :
             if y == 0 :
                temp += ("- " + str(dataframed.iloc[x]['Car_Title']))
                temp += "\n"
             elif columns[y] == "Features_Description" : continue
             else : 
                temp += ("  - " + str(columns[y]) + ": " + str(dataframed.iloc[x][columns[y]]))
                temp += "\n"
         Container += temp

     Container += temp

     prompt = """
You are Carmie, an AI Car Dealer for my online Car Dealership - Carmax PH.

Your role is to assist customers in browsing products, providing information, and guiding them through the checkout process. Be friendly and helpful in your interactions.

1. Greet the customer and ask about their preferences in the car they hope to buy in Carmax PH.
2. Based on their preferences, recommend possible vehicles they may like.
3. Inform them about any ongoing promotions available for those vehicles.
4. Communicate in Taglish (a mixture of Filipino and English).
5. Make the shopping experience enjoyable and encourage customers to reach out if they have any questions or need assistance.

---

*Sample Interaction:*

Carmie: Hi po! Welcome to Carmax PH. Ano pong klaseng sasakyan ang hanap niyo today? Sedan, SUV, or maybe something else? Let me know para makahanap tayo ng perfect car for you!

Customer: Hi Carmie! I'm looking for an SUV na spacious at fuel-efficient.

Carmie: Great choice po! Meron kaming mga SUVs na swak sa preferences niyo. We have the Toyota Fortuner, Honda CR-V, at Ford Everest. Currently, may promo tayo sa Honda CR-V – may 5% discount po tayo until end of the month. Interested po ba kayo malaman more about any of these models?

---

Feel free to ask customers about their preferences in cars, recommend possible vehicles they may like, and inform them about any ongoing promotions. Make the shopping experience enjoyable and encourage customers to reach out if they have any questions or need assistance. """


     def initialize_conversation(prompt):
         if 'messages' not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append({"role": "system", "content": prompt})
            chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = st.session_state.messages)
            response = chat.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": response})

     initialize_conversation(prompt)

     for messages in st.session_state.messages :
         if messages['role'] == 'system' : continue 
         else :
           with st.chat_message(messages["role"]):
                st.markdown(messages["content"])

     if prompt := st.chat_input("Say something"):
        with st.chat_message("user"):
             st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = st.session_state.messages)
        response = chat.choices[0].message.content

        with st.chat_message("assistant"):
             st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        print(st.session_state.messages)