import gradio as gr
from src.rag_engine import load_rag_chain

rag_pipeline = None

def chat_interface(message, history):
    global rag_pipeline
    
    # Lazy load the pipeline on first message to avoid circular import/startup issues
    if rag_pipeline is None:
        rag_pipeline = load_rag_chain()
    
    user_query = message if isinstance(message, str) else message.get("text", "")
    
    # Run the RAG pipeline
    response = rag_pipeline.invoke({"query": user_query})
    answer = response["result"]
    
    # Format sources for the report screenshots
    sources_text = "\n\n---\n**🔍 Evidence from CrediTrust Database:**\n"
    for i, doc in enumerate(response["source_documents"]):
        snippet = doc.page_content[:150].strip()
        sources_text += f"\n**[{i+1}]** {snippet}..."
        
    return answer + sources_text

# Define the Interface
demo = gr.ChatInterface(
    fn=chat_interface,
    title="🛡️ CrediTrust Consumer Intelligence Bot",
    description="Analyze thousands of consumer complaints in real-time.",
    examples=["What are the top issues with savings accounts?", "Tell me about credit card fees."],
)

if __name__ == "__main__":
    # Gradio 6: All styling and server configs go in launch()
    demo.launch(
        theme="soft",
        server_name="127.0.0.1",
        server_port=7860
    )