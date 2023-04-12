from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, QuestionAnswerPrompt, BeautifulSoupWebReader, SimpleWebPageReader, GPTListIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import os
from django.views.decorators.csrf import ensure_csrf_cookie
import whisper
import torch
import numpy as np
from scipy.io import wavfile
import wave
import librosa
import io
import soundfile as sf
import speech_recognition as sr


os.environ["OPENAI_API_KEY"] = "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"


class tourisim_model:

    index = GPTVectorStoreIndex([])

    def construct_index(self, directory_path):

        index = tourisim_model.index
        # set maximum input size
        max_input_size = 4096
        # set number of output tokens
        num_outputs = 2000
        # set maximum chunk overlap
        max_chunk_overlap = 20
        # set chunk size limit
        chunk_size_limit = 600

        # define LLM
        llm_predictor = LLMPredictor(llm=OpenAI(
            temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))
        prompt_helper = PromptHelper(
            max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

        # create documents
        documents = SimpleDirectoryReader(directory_path).load_data()
        print(documents)
        # create Index
        for document in documents:
            index.insert(document)

        return index

    def construct_index_website(self, url):

        index = tourisim_model.index
        # set maximum input size
        max_input_size = 4096
        # set number of output tokens
        num_outputs = 2000
        # set maximum chunk overlap
        max_chunk_overlap = 20
        # set chunk size limit
        chunk_size_limit = 600

        # define LLM
        llm_predictor = LLMPredictor(llm=OpenAI(
            temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))
        prompt_helper = PromptHelper(
            max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

        # create document
        documents = BeautifulSoupWebReader([]).load_data(url)

        # update index
        index = GPTVectorStoreIndex.load_from_disk('index.json')
        for document in documents:
            index.insert(document)

        index.save_to_disk('index.json')

        return index

    def ask_ai(self, query):
        index = tourisim_model.index

        # define custom QuestionAnswerPrompt
        QA_PROMPT_TMPL = (
            "We have provided context information below. \n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "these are all kinds of information about places, hotels, restaurants, events, activities and festivals happening in Saudi Arabia, as well as information about the culture and history of the area, Given this information, please answer the question in less than 30 seconds as a tour guide for Saudi Arabia, show details and recommend places and events based on the question if possible: {query_str}\n"
        )

        QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

        response = index.query(
            query, response_mode="compact", text_qa_template=QA_PROMPT)
        return (f"{response.response}")


model = tourisim_model()
index = model.construct_index('static/context')


def Index(request):
    return render(request, 'index.html')


def AskPage(request):
    return render(request, 'ask.html')


def Ask(request):
    print("in")
    if request.method == "POST":
        form = request.POST
        query = form["query"]
        response = model.ask_ai(query)
        print(response)
    else:
        response = "there's no response"
    return render(request, "ask.html", context={"query": query, "response": response})
