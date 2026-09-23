from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

database_path = 'db/chroma_db'



embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
    )

db = Chroma(
    persist_directory=database_path,
    embedding_function=embedding_model,
    collection_metadata={'hsnw:space':'cosine'}
)


while True:
    query = input("You: ")
    if query.lower() == 'exit':
        break;
    print('retrieving Answer for `',query,"`")

    retriever = db.as_retriever(search_kwargs = {'k' : 3})
    
    relavant_ans = retriever.invoke(query)

    for i,doc in enumerate(relavant_ans):
        print('Document ',i)
        print(doc.page_content)