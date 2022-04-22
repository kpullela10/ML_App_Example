import torch
import wikipedia
import transformers
import streamlit as st
from transformers import pipeline, Pipeline

def load_qa_pipeline() -> Pipeline:
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    return qa_pipeline


def load_wiki_summary(query: str) -> str:
    results = wikipedia.search(query)
    summary = wikipedia.summary(results[0], sentences = 10)
    return summary


def answer_question(pipeline: Pipeline, question: str, paragraph: str) -> dict:
    input = {
        "question": question,
        "context": paragraph
    }
    output = pipeline(input)
    return output


#Primary App Engine
if __name__ == "__main__":
    #display title and description
    st.title("Wikipedia Question Answering")
    st.write("Search topic, Ask questions, Get answers")

    #display topic input
    topic = st.text_input("SEARCH TOPIC", "")

    #display article paragraph
    article_paragraph = st.empty()

    #display question input
    question = st.text_input("QUESTION", "")

if topic:
    #load wikipedia summary of topic
    summary = load_wiki_summary(topic)

    #display article summary in paragraph
    article_paragraph.markdown(summary)

    #perform question answering
    if question != "":
        #load question-answering pipeline
        qa_pipeline = load_qa_pipeline()

        #answer query question using article summary
        result = answer_question(qa_pipeline, question, summary)
        answer = result["answer"]

        #display answer
        st.write(answer)


'''
All credit goes to Eniola Alese
Video: https://www.youtube.com/watch?v=wVF0-ZalYmk&list=WL&index=12 
Channel: https://www.youtube.com/c/eniolaa/featured
'''
