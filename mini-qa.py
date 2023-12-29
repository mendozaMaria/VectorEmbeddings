# BUILD AN AI ASSITANT PROJECT WITH PYTHON AND OPENAI

# Variables
ASTRA_DB_SECURE_BUNDLE_PATH = "YOUR_ASTRA_DB_SECURE_BUNDLE_PATH"
ASTRA_DB_APPLICATION_TOKEN = "YOUR_TOKEN_VALUE"
ASTRA_DB_CLIENT_ID = "YOUR_TOKEN"
ASTRA_DB_CLIENT_SECRET = "YOUR_ASTRA_DB_CLIENT_SECRET"
ASTRA_DB_KEYSPACE = "YOUR_KEYSPACE_NAME"
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

# Imports
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorstoreIndexCreator, VectorStoreIndexWrapper 
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

from datasets import load_dataset

# CASSANDRA CONNECTION SETUP AND AUTH, ASTRA SESSION 
cloud_config= {
    'secure_connect_bundle': ASTRA_DB_SECURE_BUNDLE_PATH
}
auth_provider = PlainTextAuthProvider(
    ASTRA_DB_CLIENT_ID,
    ASTRA_DB_CLIENT_SECRET
)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
astraSession = cluster.connect()

# LLM SETUP
llm = OpenAI(openai_api_key=OPENAI_API_KEY)
myEmbedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# CASSANDRA SETUP AND INDEX CREATION 
myCassandraVStore = Cassandra(
    embedding=myEmbedding,
    session=astraSession,
    keyspace=ASTRA_DB_KEYSPACE,
    table_name="qa_mini_demo"
)

# Load dataset, extract headlines, and store them in a Cassandra database
print("Loading dara from huggingface")
myDataset = load_dataset("Biddls/Onion_News", split="train")
headlines = myDataset["text"][:50]

print("\nGenerating embeddings and storing in AstraDB")
myCassandraVStore.add_texts(headlines)

print("Inserted %i headlines.\n" % len(headlines))

# Create index wrapper
vectorIndex = VectorStoreIndexWrapper(vectorstore=myCassandraVStore)

first_question = True
while True:
    if first_question:
        query_text = input("\nEnter your question (or type 'quit' to exit): ")
        first_question = False
    else:
        query_text = input("Enter your next question (or type 'quit' to exit): ")

    if query_text == "quit":
        break

    print("QUESTION: \"%s\""  % query_text)
    answer = vectorIndex.query(query_text, llm=llm).strip()
    print("ANSWER: \"%s\"\n" % answer)

    print("DOCUMENTS BY RELEVANCE:")
    for doc, score in myCassandraVStore.similarity_search_with_score(query_text, k=4):
        print(" %0.4f \"%s ...\"" % (score, doc.page_content[:60]))