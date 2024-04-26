# pip3 install --quiet langchain bs4 docarray tiktoken langchain_openai faiss-cpu
from langchain_community.vectorstores import DocArrayInMemorySearch #allows to create documents 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS 
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://www.kansaz.in/canada-immigration/pnp/ontario")
docs = loader.load()

embedding=OpenAIEmbeddings(openai_api_key="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")
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vectorstore = FAISS.from_documents(documents, embedding)
retriever = vectorstore.as_retriever()

# this template takes "context" and "question" to be taken in as variables
template = """Answer the question based only on the following context:
{context}

Question: {question}
Return ar the end of all responses = 'You can find more information in the link bellow: 
https://www.canada.ca/en/immigration-refugees-citizenship/services/immigrate-canada/provincial-nominees/works.htmll'
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(openai_api_key="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")
output_parser = StrOutputParser()

# Object created 
setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)

chain = setup_and_retrieval | prompt | model | output_parser

res = chain.invoke("How long is express entry PNP?")
print(res)