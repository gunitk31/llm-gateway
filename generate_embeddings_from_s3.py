import io
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from litellm import embedding
from langchain_community.vectorstores import FAISS
import faiss
import pickle

import tempfile
import os

def read_pdfs_from_s3(bucket_name, prefix=''):
    s3 = boto3.client('s3')
    documents = []
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']

                s3.download_file(bucket_name, key, f"/home/sagemaker-user/{key}")
                try:                    
                    loader = PyPDFLoader(f"/home/sagemaker-user/{key}")
                    docs = loader.load()                    
                    for doc in docs:
                        doc.metadata['source'] = key
                    documents.extend(docs)
                except Exception as e:
                    print(f"Error loading PDF {key}: {e}")
                finally:
                    try:
                        os.remove(f"/home/sagemaker-user/{key}")
                    except Exception as remove_error:
                        print(f"Error deleting file {key}: {remove_error}")
    except NoCredentialsError:
        print("AWS credentials not found.")
    except ClientError as e:
        print(f"Error accessing S3 bucket: {e}")
    return documents

def generate_titan_embedding(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = text_splitter.split_documents(documents)

    texts = [doc.page_content for doc in split_docs]

    response = embedding(
        model="amazon.titan-embed-text-v2:0",
        input=texts
    )
    
    vector_embeddings = [item["embedding"] for item in response.data]
    
    return split_docs, vector_embeddings

def main():
    bucket_name = "whitepapers-ap-south-1-795524854110"
    # Read all PDF documents from the bucket
    documents = read_pdfs_from_s3(bucket_name)

    print(f"Loaded {len(documents)} documents from S3")

    # Use FAISS vectorstore for persistence
    split_docs, embeddings = generate_titan_embedding(documents)
    print(f"Embedding response type: {type(embeddings)}")
    
    import numpy as np
    embedding_matrix = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    # Save FAISS index and documents for persistence
    faiss.write_index(index, "/home/sagemaker-user/vectordb/faiss.index")


if __name__ == "__main__":
    main()