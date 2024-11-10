from src.largelanguagemodel import TandemPartner, Critic, Translator
from src.chat import *
import streamlit as st
import pandas as pd


# Init session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vocab" not in st.session_state:
    st.session_state.vocab = {}

if "mistakes" not in st.session_state:
    st.session_state.mistakes = []

# Get user parameters
st.title("TandemLLM")
st.write("Conversational language learning made easy.")
user = st.text_input("Enter your name", "User")
col1, col2, col3 = st.columns(3)
language = col1.selectbox("Select language", LANGUAGES, index=1)
source_language = col2.selectbox("Select your native language", LANGUAGES, index=0)
level = col3.selectbox("Select level", LEVELS)
sidebar = st.sidebar
sidebar.title("Vocabulary")
sidebar.write("Add new words to your vocabulary list. You can practice them later.")
new_word = sidebar.text_input("Enter new word")
add_button = sidebar.button("Add word")
vocabulary = sidebar.container()
sidebar.divider()
sidebar.title("Corrections")
corrections = sidebar.container()

# Initialize agents
tandem = TandemPartner()
critic = Critic()
translator = Translator()

# Show chat history
show_chat_history()

# Accept new user input
if prompt := st.chat_input("Ask something in {}".format(language)):
    # user message
    st.session_state.messages.append(('human', prompt.replace('\'', '')))
    with st.chat_message(user):
        st.markdown(prompt)
    
    # agent responses
    with st.chat_message('tandem_partner'):
        response = tandem.generate_response(language, level, prompt, user, st.session_state.messages)
        tandem_response = st.write_stream(stream_output(response))
    st.session_state.messages.append(('ai', tandem_response))

    # critic responses
    critic_response = critic.generate_response(language, source_language, level, prompt)
    if critic_response['mistake_boolean']:
        correction_message = "You wrote '{}'. Write instead: {}."
        st.session_state.mistakes.append(correction_message.format(critic_response['original_message'], critic_response['correction']))

if new_word and add_button:
    if new_word not in st.session_state.vocab.keys():
        translator_response = translator.generate_response(language, source_language, new_word)
        st.session_state.vocab[translator_response['word']] = translator_response['translation']

if len(st.session_state.vocab.items()) != 0:
    vocab_df = pd.DataFrame(st.session_state.vocab.values(), index=st.session_state.vocab.keys(), columns=['translation'])
    vocabulary.dataframe(vocab_df, width=400)

for mistake in st.session_state.mistakes:
    corrections.write(mistake)
