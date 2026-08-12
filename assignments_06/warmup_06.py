from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.readers.file import PyMuPDFReader
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
import string

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# --- RAG Concepts --- 
# Concepts Q1


# Scenario A: The best approach is Retrieval-Augmented Generation (RAG) because the company has hundreds of 
# internal policy PDFs that are updated every quarter. RAG retrieves the most relevant documents at query 
# time, allowing the model to answer questions using the latest information without retraining.

# Scenario B: The best approach is fine-tuning because the company wants the model to consistently write in a 
# unique brand voice that is not common online. Since they have 3,000 examples of past writing, fine-tuning 
# can teach the model this specialized style more reliably than prompt engineering. 

# Scenario C: The best approach is prompt engineering because the analyst only needs the model to answer 
# questions about a single two-page report (simple task). The report can simply be included in the prompt, so there is 
# no need for a RAG system or fine-tuning.



# Concepts Q2


# A confidently wrong answer is more harmful than one that says "I am not sure" because the confidence level expressed in a 
# confident response can make users more likely to trust and act on the information, even if it's incorrect. 
# This can lead to incorrect actions or decisions based on the false information. A real situation(actually happened) where a 
# confident hallucination caused harm is when users asked about certain edible wild mushrooms and the AI app
# misidentified toxic wild mushrooms as safe to eat. This led to several cases of food poisoning, as people consumed the mushrooms thinking they were safe.
# The more confident the tone of the response and details the content, the more likely users are to trust it.  



# Concepts Q3

# steps = [
#     "Extract text from source documents",
#     "Split text into chunks",
#     "Convert text chunks into embeddings",
#     "Receive the user's query",
#     "Embed the user's query",
#     "Retrieve the most relevant chunks",
#     "Inject retrieved chunks into the prompt",
#     "Generate a response from the LLM",
# ]
#
# 1. Extract text from source documents - Read the text from PDFs or other documents.
# 2. Split text into chunks - Break the text into smaller sections for easier retrieval.
# 3. Convert text chunks into embeddings - Transform each chunk into a numerical vector.
# 4. Receive the user's query - Accept the user's question.
# 5. Embed the user's query - Convert the user's question into an embedding.
# 6. Retrieve the most relevant chunks - Find the document chunks most similar to the query.
# 7. Inject retrieved chunks into the prompt - Add the retrieved information to the prompt sent to the LLM.
# 8. Generate a response from the LLM - The LLM uses the prompt and retrieved context to answer the question.



def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "your", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]

# # Keyword Q1

query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

print("\n--- Keyword Retrieval Keyword Q1---")
simple_keyword_retrieval(query, documents, verbose=True)


# Add a comment explaining which document was selected and why?
# hours.txt was selected because it has the highest keyword overlap score.
# The query contains the relevant keywords "hours" and "weekends", and both
# appear in hours.txt, giving it an overlap score of 2. The other documents
# have no overlapping keywords with the query.


# Keyword Q2

query = "Do you have anything without caffeine?"

print("\n--- Keyword Retrieval Keyword Q2 ---")
simple_keyword_retrieval(query, documents, verbose=True)


# No document was selected because there were no overlapping keywords between the query and any of the 
# documents. The function returned "None found" as the best match.

# Keyword RAG did NOT get this right. The menu.txt document contains information about drinks, which is 
# relevant to the question, but it does not use the exact keywords "caffeine" or "without." Because keyword 
# retrieval depends on exact word overlap, it failed to recognize the semantic relationship.

# Semantic RAG search would do better here because it would search for the meaning instead exactkeywords 
# matching in the query and recognized that "without caffeine" relates to ingredient items, even if the 
# exact keywords don't match. It will then retrieve documents that discuss ingredients, which could 
# potentially include information about caffeine content.


# Keyword Q3

# I predict it will return "None found" because the query "How do I sign up for rewards?" does not 
# share keywords with any of the documents.

query = "How do I sign up for rewards?"

print("\n--- Keyword Retrieval Keyword Q3 ---")
simple_keyword_retrieval(query, documents, verbose=True)

# --------------------------------------------------
# Reflection:
#
# Yes, my prediction was correct. The function returned "None found" because there were no overlapping 
# keywords between the query and any of the documents. The result is interesting because loyalty.txt 
# is clearly relevant to the question, but it uses different terminology: "loyalty program" instead of
# "rewards." Keyword retrieval cannot recognize that these terms have similar meanings.
# --------------------------------------------------

# --- Semantic RAG Concepts --- 

# Semantic Q1

# What is a vector embedding? (1-2 sentences)
# A vector embedding is a list of numbers that represents the meaning or characteristics of data, such as texts,images, 
# and audio. 

# Two text chunks have cosine similarity scores of 0.85 and 0.30 with a given query. 
# The chunk with a cosine similarity score of 0.85 is more relevant because it is more similar to the query than the 
# chunk with a score of 0.30. A higher score means the two vectors have a closer relationship in meaning or direction or
# greater semantic similarity. 

# Why can semantic search find a relevant chunk even when none of the exact words from the query appear in the chunk?
# Semantic search can find relevant chunks even when they don't contain the exact words from the query because it 
# compares the meaning of the query and the text using vector embeddings rather than relying only on keyword matches. 

# Semantic Q2

# | Feature                    | Keyword RAG                       | Semantic RAG                    |
# |----------------------------|-----------------------------------|---------------------------------|
# | What is compared?          | Exact word overlap                | query /chunk embeddings         |
# | What is retrieved?         | Full document                     | Similar chunks                  |
# | Can it handle synonyms?    | No                                | yes                             |
# | Storage format             | Plain text dictionary             | Vector Embeddings               |
# | Relevance score            | Number of overlapping keywords    | Cosine similarity               |


# --- LlamaIndex --- 

# LlamaIndex Q1

# Load documents directly from PDFs in the folder
brightleaf_path = "../../python-200-v1/lessons/06_AI_augmentation/resources/brightleaf_pdfs"
docs = SimpleDirectoryReader(brightleaf_path, file_extractor={".pdf": PyMuPDFReader()}).load_data()

# Build a vector index automatically (handles chunking + embeddings)
index = VectorStoreIndex.from_documents(docs)

print(type(index._vector_store).__name__)

query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)

# --------------------------------------------------
# Reflection for employee benefits:
#
# Do the retrieved chunks look relevant?
# Yes. The highest-scoring chunk is relevant to employee benefits because it
# contains information about the benefits BrightLeaf offers. The other two
# retrieved chunks are less relevant because they discuss renewable energy
# rather than employee benefits.
#
# Does the model's response sound confident and specific?
# The model's response sounds confident and specific. It provides a direct
# answer about BrightLeaf's employee benefits without using uncertain language
# such as "I'm not sure."
#
# Did anything unexpected get retrieved?
# Yes. Some of the retrieved chunks discuss renewable energy, which is not
# directly related to employee benefits.

# Reflection for security policies:
#
# Do the retrieved chunks look relevant?
# Yes. The highest-scoring chunk is relevant to BrightLeaf's security policies,
# while the other retrieved chunks are less relevant because they discuss
# renewable energy.
#
# Does the model's response sound confident and specific?
# The model's response sounds confident and specific. It gives a direct answer
# without expressing uncertainty.
#
# Did anything unexpected get retrieved?
# Yes. Some of the retrieved chunks discuss renewable energy, which was
# unexpected because the question asks about BrightLeaf's security policies.
# --------------------------------------------------

# LlamaIndex Q2

k_list = [1, 5]

for k_ in k_list:
    query_engine = index.as_query_engine(similarity_top_k=k_)
    question = "What employee benefits does BrightLeaf offer?"
    response = query_engine.query(question)

    print(f"\n--- similarity_top_k={k_} ---")
    print(f"Q: {question}")
    print(f"A: {response}")

    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)


# --------------------------------------------------
# Reflection:
#
# Add a comment explaining how the response changed (if at all) and whether more retrieved context is always better.

# Increasing similarity_top_k from 1 to 5 did not significantly change the response because the first retrieved 
# node already contained the relevant information needed to answer the question. The additional nodes did not add
# much useful information. More retrieved context is not always better because irrelevant chunks can add noise and 
# potentially make the response less focused or accurate.

# --------------------------------------------------

# LlamaIndex Q3

k_ = 3
query_engine = index.as_query_engine(similarity_top_k=k_)
question = "What are the calories for each menu item?"
response = query_engine.query(question)

print(f"\n--- similarity_top_k={k_} ---")
print(f"Q: {question}")
print(f"A: {response}")


for node_with_score in response.source_nodes:
    print(f"Node ID: {node_with_score.node.node_id}")
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node_with_score.node.get_content()}")
    print("-" * 30)


# --------------------------------------------------
# Reflection:
#
# Add a comment explaining what you expected, what actually happened, and what you would change about the 
# system to handle this kind of query better.

# I expected the system to find no relevant context because the documents do not contain information about
# menu item calories. That is what actually happened. The system retrieved some nodes, but they were not
# relevant to the query. To handle this type of query better, I would consider adding a filtering mechanism so 
# that low-relevance nodes are not passed to the model. This could help the system recognize when the retrieved 
# documents do not contain enough information to answer the question instead of returning unrelated source nodes.

# --------------------------------------------------

# LlamaIndex Q4

# Create Judge LLM
llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

# Define evaluator
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

# --------------------------------------------------
# Query 1: Question about information in the documents
# --------------------------------------------------

# Get response to query
q = "What employee benefits does BrightLeaf offer?"
response = query_engine.query(q)

# Evaluate faithfulness and relevancy
faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)

print("Faithfulness Evaluation: " + str(faithfulness_result.score))
print("Relevancy Result: " + str(relevancy_result.score))

# --------------------------------------------------
# Query 2: Question about information not in the documents
# --------------------------------------------------

q2 = "What store location do you have in Mars?"
response2 = query_engine.query(q2)

faithfulness_result2 = faithfulness_evaluator.evaluate_response(query=q2, response=response2)

relevancy_result2 = relevancy_evaluator.evaluate_response(query=q2, response=response2)

print("Faithfulness Score (Query 2):", faithfulness_result2.score)
print("Relevancy Score (Query 2):", relevancy_result2.score)

# --------------------------------------------------
# Reflection:
#
# What does a faithfulness score of 1.0 mean? What would a score of 0.0 indicate?
# A faithfulness score of 1.0 means the response is fully supported by the retrieved context. A score of
# 0.0 indicates that the response is not supported by the retrieved context.

# What does a relevancy score measure, and how is it different from faithfulness?
# Relevancy score measures how relevant the model's response is to the user's query, while 
# faithfulness measures how accurately the response reflects the information retrieved from the documents. 
# A high relevancy score indicates that the response addresses or directly answers the user's question, while a 
# high faithfulness score indicates that the response is based on accurate information from the retrieved context.

# Did the scores change between your two queries? If so, why do you think that happened?
# Yes, the scores changed between the two queries. The first query about employee
# benefits had high faithfulness and relevancy scores because the information
# was present in the documents, allowing the model to provide an accurate and
# relevant response.

# The second query about a store location on Mars received a lower relevancy
# score because the BrightLeaf documents did not contain information about a
# store location on Mars. Faithfulness remained high because the model avoided
# making unsupported claims and stated that the information was not available
# in the retrieved context.

# What is the "LLM-as-a-judge" approach, and why is it used for RAG evaluation instead of a 
# simple accuracy metric?

# LLM-as-a-judge approach is a method where an LLM is used to evaluate the performance of another
# LLM's response according to criteria such as faithfulness and relevancy. It is used for 
# RAG evaluation instead of a simple accuracy metric because answers are open-ended and may not have a
# a single correct answer and there are many ways to express the same information. Therefore,
# an accuracy metric is harder to apply as there is no single exact and correct answer to compare against.