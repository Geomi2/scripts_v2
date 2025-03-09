import argparse
import numpy as np
import ollama
from typing import List, Dict
from langchain.schema import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings


BOOKS_PATH = "/Users/chara/Documents/thesis/scripts_v2/books/"
MODEL = "nomic-embed-text"
CHROMA_PATH = f"./{MODEL}_db"

embedding_function = OllamaEmbeddings(model=MODEL)
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def check_chroma_integrity():
    collection_names = vectorstore._client.list_collections()
    book_titles = [c.name for c in collection_names]

    for collection_name in book_titles:
        print(f"Checking collection: {collection_name}")

        collection = vectorstore._client.get_collection(collection_name)
        data = collection.get(include=["documents", "embeddings"])  

        chunks = data.get("documents") 
        embeddings = data.get("embeddings")  

        null_chunks = []
        null_embeddings = []

        for chunk in chunks:
            if chunk is None:
                null_chunks.append(chunk)

        for chunk, embedding in zip(chunks, embeddings):
            if embedding is None:
                null_embeddings.append(chunk)

    if null_chunks:
        print(f"⚠️ Collection '{collection_name}' has {len(null_chunks)} embeddings without chunks.\n")

    if null_embeddings:
        print(f"⚠️ Collection '{collection_name}' has {len(null_embeddings)} chunks without embeddings.\n")

    if not null_embeddings and not null_chunks:
        print("\n✅ No issues found. The database is consistent.")

def book_choice():
    collection_names = vectorstore._client.list_collections()
    book_titles = [c.name for c in collection_names]

    if not book_titles:
        print("❌ No books found in the database!")
        exit()

    print("\n📚 Available books:")
    for title in book_titles:
        print(f"- {title}")     

    while True:
        selected_book = input("\n🔹 Enter the book title: ").strip()
        if selected_book in book_titles:
            break
        print("⚠️ Invalid title, please try again.")

    print(f"\n✅ You chose: {selected_book}")
    return selected_book


def query_rag(query_text: str, book_for_qa):

    collection = vectorstore._client.get_collection(book_for_qa)

    data = collection.get(include=["documents", "embeddings"])
    metadatas = collection.get(include=["metadatas"])

    raw_documents = data.get("documents", [])
    documents = [Document(page_content=doc) for doc in raw_documents if isinstance(doc, str)]  # Έλεγχος για έγκυρα strings
    embeddings = np.array(data.get("embeddings", []))  # Μετατροπή σε numpy array

    # Debug checks
    #print(f"Collection: {collection}")
    #print(f"Data: {data}")
    #print(f"Documents: {documents}")
    #print(f"Embeddings: {embeddings}")

    if not documents:
        print("❌ No valid documents found for this book!")
        return

    if not collection:
        return "Book not found in the database."

    query_vector = embedding_function.embed_documents(query_text)   # Embed the query text

    # Search the DB.
    results = vectorstore.similarity_search_with_score(query_vector, k=5, filter=List[Dict[documents: str, embeddings: int]], where_document=book_for_qa)  
    # TypeError: Parameters to generic types must be types. Got slice([Document(metadata={}, page_content='MONOPOLY \nProperty Trading Game from Parker Brothers" \n.
    if not results:
        print("❌ No results found for the query!")
        return

    print(results)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    response_text = MODEL.invoke(prompt)
    sources = [doc.metadata.get("id", None) for doc in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


def main():
    check_chroma_integrity()
    book_for_qa = book_choice()

    query_text = input("\nAsk a question: ")
    query_rag(query_text, book_for_qa)
 

if __name__ == "__main__":
    main()