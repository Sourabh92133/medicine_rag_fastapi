from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

loader = CSVLoader(
    file_path="Medicine_database.csv",
    source_column="product_manufactured",
    encoding="ISO-8859-1"
)

documents = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="vector_database"
)

print("Vector DB created successfully")
