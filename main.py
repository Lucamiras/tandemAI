from langchain_helper import TandemPartner, Critic, Translator
import streamlit as st
import pandas as pd
import time


# Init session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vocab" not in st.session_state:
    st.session_state.vocab = {}

if "mistakes" not in st.session_state:
    st.session_state.mistakes = []

# Get user parameters
st.title("Tandem AI")
st.write("Welcome to the Tandem AI chatbot. You can practice your language skills here.")
user = st.text_input("Enter your name", "User")
col1, col2, col3 = st.columns(3)
language = col1.selectbox("Select language", ["Hungarian", "Danish", "German", "Italian", "Spanish", "French", "Portuguese", "Dutch", "Swedish", "Norwegian", "Finnish"])
source_language = col2.selectbox("Select your native language", ["English", "Hungarian", "Danish", "German", "Italian", "Spanish", "French", "Portuguese", "Dutch", "Swedish", "Norwegian", "Finnish"])
level = col3.selectbox("Select level", ["Beginner", "Intermediate", "Advanced"])
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
for message in st.session_state.messages:
    with st.chat_message(message[0]):
        st.markdown(message[1])

# Stream response
def response_generator(prompt, agent):
    response = agent.generate_response(language, level, prompt, user, st.session_state.messages)
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

# Accept new user input
if prompt := st.chat_input("Ask something in {}".format(language)):
    # user message
    st.session_state.messages.append(('human', prompt.replace('\'', '')))
    with st.chat_message(user):
        st.markdown(prompt)
    
    # agent responses
    with st.chat_message('tandem_partner'):
        tandem_response = st.write_stream(response_generator(prompt, tandem))
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




    


