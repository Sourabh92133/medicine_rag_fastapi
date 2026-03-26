import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


df=pd.read_csv("Medicine_database.csv",encoding="ISO-8859-1")  # load csv into data frame
documents=[]

# we are restructuring the data so that each row becomes semantically complete document
for i,(_,row) in enumerate(df.iterrows()):
    text=f"""
    Product Name: {row.get("product_name","")}
    Salt composition: {row.get("salt_composition","")}
    Description: {row.get("medicine_desc","")}
    Price: ₹{row.get("product_price","")}
    Manufacturer: {row.get("product_manufactured","")}
    Category: {row.get("sub_category","")}
    """

    documents.append(Document(page_content=text))   # convert into langchain format

## now need of this because i have converted the data into document list so that each row represents meaningful adn complete document

# loader = CSVLoader(
#     file_path="Medicine_database.csv",
#     source_column="product_manufactured",
#     encoding="ISO-8859-1"
# )

# documents = loader.load()

# text_splitter = CharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
# )
#
# chunks = text_splitter.split_documents(documents)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    persist_directory="vector_database"
)

print("Vector DB created successfully")
