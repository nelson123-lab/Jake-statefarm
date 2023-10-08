import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
# FAISS runs locally saves the embeddings on the local machine.
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

# To read pdf docs.
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        # Inorder to extract each page contents.
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# To generate text chunks from the text data.
def get_text_chunks(text):
    """Chunk overlap is helping in taking the text not to split into half. Takes care of
    the left over when next set of characters are considered"""
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len)
    chunks = text_splitter.split_text(text)
    return chunks

# To generate the vector database from the text chunks.
def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name = 'hkunlp/instructor-xl')
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vectorstore.as_retriever(),
        memory = memory
    )
    return conversation_chain


def main():
    load_dotenv()
    st.set_page_config(page_title = "Chat with Jake", page_icon=":books")

    st.write(css, unsafe_allow_html=True)
    st.header("Chat with Jake")
    st.text_input('What should I guide you with?')

    st.write(user_template.replace("{{MSG}}", "Hello Jake"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello Nelson"), unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your documents", accept_multiple_files = True)
        if st.button("Process"):
            # To show a processing circle instead of staying still.
            with st.spinner("Processing"):
                # Get pdf
                raw_text = get_pdf_text(pdf_docs)
                
                # get the text chunks
                # Returns a list of chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)
                
                # create vector store.
                vector_store = get_vector_store(text_chunks)

                # Create conversation chain
                # Takes history of the conversation and return replies to it.
                # Session_state is used to not reinitialize the variables each time.
                st.seesion_state.converstaion = get_conversation_chain(vector_store)


if __name__ == '__main__':
    main()