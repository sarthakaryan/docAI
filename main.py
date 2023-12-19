# app.py
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, T5ForConditionalGeneration, T5Tokenizer, pipeline
from bs4 import BeautifulSoup
import requests
import re
import streamlit as st
import spacy
import os
os.environ['TRANSFORMERS_CACHE'] = "D://Softwares//Python//Models Cache"

nlp = spacy.load("en_core_web_sm")

def getLinks(text) :
    text = "+".join([token.text for token in nlp(text) if token.pos_ in ['NOUN','PROPN']])
    url = "https://medicalsciences.stackexchange.com/search?q={}+answers%3A1"
    soup = BeautifulSoup(requests.get(url.format(text)).text, 'html.parser')
    L = soup.find(class_="js-post-summaries").find_all(class_="s-link")
    links = list(map(lambda x : "https://medicalsciences.stackexchange.com" + x['href'], L))
    return links

def getData(links) :
    texts = []
    for i in links :
        temp = BeautifulSoup(requests.get(i).text, 'html.parser')
        parent = temp.find(class_="answercell")
        child = parent.find(class_="js-post-body")
        texts.append(re.sub(" +"," ",re.sub("\n"," ",child.text)).strip())
    data = " ".join(texts)
    return data

def generate_text_completion(model,tokenizer,seed_text, context, max_length=50):
    input_text = f"{context} Complete the following: '{seed_text}'"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    output = model.generate(input_ids, max_length=max_length, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

qa_model_name = "distilbert-base-cased-distilled-squad"
qa_tokenizer = DistilBertTokenizer.from_pretrained(qa_model_name)
qa_model = DistilBertForQuestionAnswering.from_pretrained(qa_model_name)
qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


st.title("DocAI")
# Streamlit UI
user_input = st.text_input("Enter your message:")
send_button = st.button("Send")

hide_streamlit_style = """
            <style>
            .reportview-container {
                margin-top: -2em;
            }
            #MainMenu {visibility: hidden;}
            .stDeployButton {display:none;}
            footer {visibility: hidden;}
            #stDecoration {display:none;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if send_button :
    st.write("**DocAI searching in ...**")
    links = getLinks(user_input)
    text = " ".join([token.text for token in nlp(user_input) if token.pos_ in ['NOUN','PROPN','VERB']])
    for i in links:
        st.write(i)
    st.write("**Getting the data ...**")
    data = getData(links)
    st.write("Done ✅")
    st.write("**Getting attention words ...**")
    answer = qa_pipeline(question=text, context=data)["answer"]
    st.write("Done ✅")
    st.write("**Generating Text! Please wait for a while ...**  ")
    message = generate_text_completion(model, tokenizer, answer, data, max_length=200)
    sentences = [sent.text for sent in nlp(message).sents][:5]
    st.write(" ".join(sentences))


