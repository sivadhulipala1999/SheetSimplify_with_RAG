from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy


EMBEDDING_MODEL_NAME = "thenlper/gte-small"


def prep_vectorstore_csv(filepath):
    """Take the CSV data and load it into the FAISS vector store using HuggingFace embeddings"""
    loader = CSVLoader(file_path=filepath, encoding="utf-8", csv_args={
        'delimiter': ','})
    data = loader.load()
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},)
    vectorstore = FAISS.from_documents(
        data, embeddings, distance_strategy=DistanceStrategy.COSINE)
    return vectorstore


######## Personal Notes ########
# Using cosine similarity here since Euclidean will not be accurate in multi-dimensional spaces and with sparse vectors
