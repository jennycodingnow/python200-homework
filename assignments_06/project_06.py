from dotenv import load_dotenv
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.readers.file import PyMuPDFReader


# --------------------------------------------------
# Step 1: Setup
# --------------------------------------------------
print(f"\nStep 1:\n")

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

docs_dir = Path("../../python-200-v1/lessons/06_AI_augmentation/resources/groundwork_docs/")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"


# --------------------------------------------------
# Step 2: Load the Documents
# --------------------------------------------------

print(f"\nStep 2:\n")

docs = SimpleDirectoryReader(docs_dir, file_extractor={".pdf": PyMuPDFReader()}).load_data()

print(f"\nDocuments loaded: {len(docs)}\n")
print(f"\nDocument names:\n")
for doc in docs:
    print(f" - {doc.metadata['file_name']}")


# --------------------------------------------------
# Step 3: Build the Index and Query Engine
# --------------------------------------------------

print(f"\nStep 3:\n")

index = VectorStoreIndex.from_documents(docs)

query_engine = index.as_query_engine(similarity_top_k=3)

print("Index built successfully. Ready to answer questions.")

# --------------------------------------------------
# Step 4: Query the Assistant
# --------------------------------------------------

print(f"\nStep 4:\n")

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    print("\n--------------------------------\n")
    node_with_score = response.source_nodes[0]
    node = node_with_score.node

    print(f"Document: {node.metadata.get('file_name')}")
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node.get_content()[:200]}...")

# --------------------------------------------------
# Reflection:
#
# Add a comment reflecting on the responses: 
# Did the assistant sound confident and accurate? 
# Yes, the assistant sounded confident and accurate in its responses in this case. The answers were relevant 
# to the questions asked and were supported by information from the source documents. The assistant was able 
# to retrieve specific details about Groundwork's hours, dairy-free milk options, loyalty program, company history,
# and catering/wholesale services.

# Did any of the answers surprise you?
# No, none of the answers surprised me. The responses were consistent with what I expected based on the questions asked 
# and the information available in the source documents. And also the answers were relevant and aligned with the context 
# of the questions.

# --------------------------------------------------


# --------------------------------------------------
# Step 5: Find a Failure
# --------------------------------------------------

print(f"\nStep 5:\n")

q = "Are there vegan options available?"
print(f"\nQ: {q}")
response = query_engine.query(q)
print("A:", response)

for node_with_score in response.source_nodes[:3]:
    node = node_with_score.node
    document_name = node.metadata.get('file_name')

    print(f"Document: {document_name}")
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node.get_content()[:200]}...")
    print("-" * 30)


# --------------------------------------------------
# Reflection:
# # Add a comment reflecting on the responses:

# What you asked and why you expected it to be hard
# I asked about vegan options availablilty because I expected it to be hard for the assistant 
# to provide an accurate answer. The source documents may not have contained information about vegan options, 
# making it challenging for the assistant to retrieve relevant information.

# What went wrong — wrong retrieval, missing information, the model guessed anyway?
# The retrieval was wrong because the source documents did not contain any information about vegan options. 
# The system retrieved some nodes, but they were not relevant to the query.
# The model likely guessed or provided a default response since it couldn't find relevant information in the 
# documents.


# When the retrieval failed, did the model's tone change — did it become less certain, or did it still sound 
# confident even when it was wrong? What does this suggest about trusting AI-generated responses?
# The model's tone did not change significantly, and it still sounded confident even when it was wrong. This 
# suggests that we should be cautious when relying on AI-generated responses, as the model may appear confident 
# in its answers even when they are not based on accurate information from the source documents.


# What you would change about the system to improve it?
# I would improve the system by adding a stronger mechanism for recognizing when the retrieved documents 
# do not contain enough information to answer a question. The assistant should be able to say that the 
# information was not found rather than confidently generating an unsupported answer. I would also add 
# faithfulness and relevancy evaluation metrics to measure whether the response is supported by the retrieved 
# documents and whether the retrieved documents are relevant to the question. These changes would help reduce 
# hallucinations and make the RAG system more reliable.

# --------------------------------------------------

# --------------------------------------------------
# Step 6: Reflection
# --------------------------------------------------

# Add a comment block to answer the following questions:

# 1. The lesson built semantic RAG manually — chunking, embedding, and indexing took many lines of code. How many lines 
# did the equivalent LlamaIndex implementation take in your project? What does that tell you about the value of using 
# a framework?

# The equivalent LlamaIndex implementation took approximately 23 lines of code more or less when counting the 
# the number of questions asked and print statements, compared with many more lines in the manual implementation. 
# This shows that frameworks like LlamaIndex make it easier and faster to build RAG applications because they 
# handle many of the complicated steps for you, such as chunking, embedding, and indexing.


# 2. You have now built a system that answers questions from real documents. Describe a different use case — 
# not a coffee shop — where this approach would add genuine value to a business or organization.

# A useful business use case would be an internal HR assistant. A company could load employee handbooks, 
# benefits documents, leave policies, and workplace procedures into a RAG system. Employees could then 
# ask questions in plain language and quickly retrieve the relevant company policies without having to search through
# many documents manually.


# 3. What is one failure mode that RAG cannot fully prevent, even when retrieval is working correctly?

# One failure mode that RAG cannot fully prevent is the issue of hallucination, where the model generates
# information that is not present in the source documents. Even if the retrieval process is working correctly, 
# the model may still produce inaccurate or misleading responses based on its training data or biases. 
