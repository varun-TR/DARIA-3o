import os
import json
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter

def main():
        

    # Set HuggingFace API Token
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


    def extract_json_content(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            combined_text = []
            for key, value in data.items():
                for entry in value:
                    if isinstance(entry, str):
                        combined_text.append(entry)
                    elif isinstance(entry, dict) and "caption" in entry:
                        combined_text.append(entry["caption"])
            return "\n".join(combined_text).strip()
        except Exception as e:
            st.error(f"Error reading JSON file: {e}")
            return None

    def process_content(content):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(content)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        return knowledge_base

    def load_model(repo_id, temperature=0.1, max_length=800):
        return HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={
                "temperature": temperature,
                "max_length": max_length,
            }
        )

    def generate_answer(llm, context, user_question):
        prompt = (
            f"Use the following context to answer the question strictly based on the provided information. "
            f"If the answer is not present in the context, respond with 'I don't know.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {user_question}\n"
            f"Answer:"
        )
        full_answer = llm(prompt).strip()
        first_answer = full_answer.split("Answer:")[1].strip().split("\n")[0]
        return first_answer

    # Main Page Content
    st.title("DARIA-3o: Chatbot for InfoTunnel")

    # JSON file path
    json_path = "/Users/saivaruntanjoreraghavendra/Documents/json/scraped_content.json"

    # Load JSON content
    json_content = extract_json_content(json_path)
    if not json_content:
        st.error("Failed to load JSON file. Please check the file path.")
    else:
        knowledge_base = process_content(json_content)
        st.success("Knowledge Base loaded successfully!")

        user_question = st.text_input("Ask a question about the content:")
        if user_question:
            with st.spinner("Generating answers..."):
                # Retrieve documents once
                retrieved_docs = knowledge_base.similarity_search(user_question, k=5)
                if retrieved_docs:
                    context = "\n".join([doc.page_content for doc in retrieved_docs])

                    # Generate answers using both models
                    microsoft_llm = load_model(repo_id="microsoft/Phi-3.5-mini-instruct")
                    microsoft_answer = generate_answer(microsoft_llm, context, user_question)

                    llama_llm = load_model(repo_id="meta-llama/Meta-Llama-3-8B-Instruct")
                    llama_answer = generate_answer(llama_llm, context, user_question)

                    st.subheader("Answer from Microsoft Phi2")
                    st.write(microsoft_answer)

                    st.subheader("Answer from Meta LLaMA")
                    st.write(llama_answer)

                    st.subheader("Sources")
                    with st.expander("View Sources"):
                        for doc in retrieved_docs:
                            st.write(f"- {doc.page_content[:200]}...")
                else:
                    st.write("No relevant documents were found in the knowledge base.")
