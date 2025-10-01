from main import retrieve_relevant_chunks, answer_question, chunks, chunk_embeddings, embedding_model

print("\nAI Assistant is ready! Type 'quit' to exit.")
while True:
    user_query = input("\nAsk a question: ")
    if user_query.lower() == "quit":
        break
    relevant_chunks = retrieve_relevant_chunks(user_query, chunks, chunk_embeddings, embedding_model)
    answer = answer_question(user_query, relevant_chunks)
    print("\nRelevant Chunks:")
    for idx, chunk in enumerate(relevant_chunks, 1):
        print(f"{idx}. {chunk}")
    print("\nAnswer:")
    print(answer)
