import os

#os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.memory import ConversationBufferMemory
import streamlit as st
from streamlit_chat import message
from streamlit_extras.add_vertical_space import add_vertical_space

st.set_page_config(page_title="ACS Cancer Bot", page_icon=":hospital:")

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferMemory(memory_key='chat_history', output_key='answer', return_messages=True)


persist_directory = "acs_db"
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding)
    


retriever = vectordb.as_retriever(search_kwargs={"k": 10})
qa_chain = ConversationalRetrievalChain.from_llm(llm=OpenAI(model_name = 'gpt-3.5-turbo-16k', max_tokens=5000),
                                                 memory=st.session_state.buffer_memory,
                                                 retriever=retriever, 
                                                 return_source_documents=True)




# Sidebar contents
with st.sidebar:
    st.title('ACS Cancer Bot 🤗💬')
    st.markdown('''This is a chatbot that can answer questions about cancer.
                It gets its info from up-to-date resources on the American Cancer Society's website.
                Please ask it questions you want to know about common cancers, treatments, and more.''')
    add_vertical_space(5)
    

# Generate empty lists for generated and past.
## generated stores AI generated responses
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["I'm ACS Bot, How may I help you?"]
## past stores User's questions
if 'requests' not in st.session_state:
    st.session_state['requests'] = ['Hi!']

# Layout of input/response containers
input_container = st.container()
response_container = st.container()

# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text
## Applying the user input box
with input_container:
    user_input = get_text()

# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(request):
    response = qa_chain(request)
    return response

## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if user_input:
        response = generate_response(user_input)['answer']
        st.session_state.requests.append(user_input)
        st.session_state.responses.append(response)
        
    if st.session_state['responses']:
        memory_len = len(st.session_state['responses'])
        for i in range(memory_len):
            message(st.session_state['requests'][memory_len - i - 1], is_user=True, key=str(i) + '_user', avatar_style='thumbs')
            message(st.session_state['responses'][memory_len - i - 1], key=str(i), avatar_style='bottts-neutral')


