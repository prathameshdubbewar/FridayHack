import streamlit as st
import os
import json
import tempfile
from typing import Dict, List, Tuple, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import httpx
import warnings
warnings.filterwarnings("ignore")

# Document processing imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain.schema import Document
import PyPDF2
from io import BytesIO
import base64
import hashlib

# Configure SSL
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize session state
def init_session_state():
    default_states = {
        "messages": [],
        "api_key": "",
        "repair_jobs": [],
        "uploaded_documents": [],
        "vector_store": None,
        "document_chunks": [],
        "document_metadata": {},
        "current_context": "",
        "show_reasoning": True,
        "processing_log": []
    }
    
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize LLM
@st.cache_resource
def initialize_llm(api_key: str):
    client = httpx.Client(verify=False)
    llm = ChatOpenAI(
        base_url="https://genailab.tcs.in",
        model="azure_ai/genailab-maas-DeepSeek-V3-0324",
        api_key=api_key,
        http_client=client,
        temperature=0.1,  # Lower temperature for more consistent responses
        max_tokens=4000
    )
    return llm

# Initialize embeddings
@st.cache_resource
def initialize_embeddings(api_key: str):
    # Note: You might need to adjust this based on your embedding service
    # Using same base URL for embeddings - adjust if needed
    return OpenAIEmbeddings(
        openai_api_key=api_key,
        openai_api_base="https://genailab.tcs.in",
        model="text-embedding-ada-002"
    )

# Document processing functions
def extract_text_from_pdf(file_bytes, filename):
    """Extract text from PDF with page tracking"""
    text = ""
    metadata = []
    
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                metadata.append({
                    "page": page_num + 1,
                    "source": filename,
                    "type": "pdf"
                })
        
        if not text.strip():
            return "", [], "No text could be extracted from PDF"
            
        return text, metadata, "Success"
        
    except Exception as e:
        return "", [], f"Error processing PDF: {str(e)}"

def extract_text_from_txt(file_bytes, filename):
    """Extract text from TXT file"""
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                text = file_bytes.decode(encoding)
                metadata = [{
                    "page": 1,
                    "source": filename,
                    "type": "txt",
                    "encoding": encoding
                }]
                return text, metadata, "Success"
            except UnicodeDecodeError:
                continue
        
        return "", [], "Could not decode text file with common encodings"
        
    except Exception as e:
        return "", [], f"Error reading text file: {str(e)}"

def process_document(file, filename):
    """Process uploaded document"""
    file_bytes = file.read()
    file_extension = filename.split(".")[-1].lower()
    
    if file_extension == "pdf":
        return extract_text_from_pdf(file_bytes, filename)
    elif file_extension == "txt":
        return extract_text_from_txt(file_bytes, filename)
    else:
        return "", [], f"Unsupported file type: {file_extension}"

def chunk_document(text, metadata, chunk_size=1000, chunk_overlap=200):
    """Split document into chunks for vector storage"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Create chunks with metadata
    chunks = text_splitter.create_documents([text])
    
    # Add metadata to each chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata = {
            "chunk_id": i,
            "source": metadata[0]["source"] if metadata else "unknown",
            "page": metadata[0]["page"] if metadata else 1,
            "type": metadata[0]["type"] if metadata else "unknown",
            "total_chunks": len(chunks)
        }
    
    return chunks

def create_vector_store(chunks, embeddings, api_key):
    """Create vector store from document chunks"""
    if not chunks:
        return None
    
    try:
        # Create temporary directory for Chroma
        persist_dir = tempfile.mkdtemp()
        
        # Create vector store
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        
        # Store chunk information
        st.session_state.document_chunks = chunks
        
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def retrieve_relevant_chunks(query, vector_store, k=5):
    """Retrieve relevant document chunks for a query"""
    if not vector_store:
        return []
    
    try:
        # Perform similarity search
        results = vector_store.similarity_search_with_relevance_scores(query, k=k)
        
        # Format results with citations
        formatted_results = []
        for i, (doc, score) in enumerate(results):
            formatted_results.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 1),
                "score": score,
                "chunk_id": doc.metadata.get("chunk_id", i)
            })
        
        return formatted_results
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return []

def generate_citation_prompt(query, retrieved_chunks):
    """Generate prompt with citations"""
    context_parts = []
    
    for i, chunk in enumerate(retrieved_chunks):
        context_parts.append(
            f"[Source {i+1}]: From {chunk['source']}, Page {chunk['page']}\n"
            f"Content: {chunk['content']}\n"
            f"Relevance Score: {chunk['score']:.3f}\n"
        )
    
    context = "\n".join(context_parts)
    
    prompt = f"""
    You are an expert repair technician AI assistant. You MUST use ONLY the provided document context to answer questions.
    If the context doesn't contain relevant information, say so explicitly.
    
    DOCUMENT CONTEXT WITH CITATIONS:
    {context}
    
    USER QUERY: {query}
    
    INSTRUCTIONS:
    1. FIRST, analyze which sources are relevant to the query
    2. SECOND, extract specific information from relevant sources
    3. THIRD, formulate your answer using ONLY information from cited sources
    4. FINALLY, provide citations in format: [Source X, Page Y] for each claim
    
    THINKING PROCESS (show your reasoning):
    1. Query analysis: What is being asked?
    2. Source relevance: Which sources contain relevant information?
    3. Information extraction: What specific facts are found?
    4. Answer formulation: How to answer based on extracted facts?
    
    ANSWER FORMAT:
    ## Thinking Process
    [Your step-by-step reasoning here]
    
    ## Answer
    [Your answer with citations]
    
    ## Sources Used
    - List of sources referenced
    """
    
    return prompt, context

def analyze_with_rag(llm, query, vector_store):
    """Analyze query using RAG with reasoning"""
    # Step 1: Retrieve relevant chunks
    retrieved_chunks = retrieve_relevant_chunks(query, vector_store, k=5)
    
    if not retrieved_chunks:
        return {
            "answer": "No relevant information found in uploaded documents.",
            "thinking": "No documents were retrieved for this query.",
            "sources": [],
            "context": ""
        }
    
    # Step 2: Generate prompt with citations
    prompt, context = generate_citation_prompt(query, retrieved_chunks)
    
    # Step 3: Get LLM response
    try:
        response = llm.invoke(prompt)
        content = response.content
        
        # Parse response to extract thinking and answer
        thinking = ""
        answer = ""
        sources = []
        
        # Try to parse structured response
        if "## Thinking Process" in content and "## Answer" in content:
            parts = content.split("## Answer")
            thinking = parts[0].replace("## Thinking Process", "").strip()
            answer = "## Answer" + parts[1] if len(parts) > 1 else content
        else:
            thinking = "LLM did not provide structured thinking process."
            answer = content
        
        # Extract sources from chunks
        sources = [
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "relevance": chunk["score"],
                "content_preview": chunk["content"][:100] + "..."
            }
            for chunk in retrieved_chunks
        ]
        
        return {
            "answer": answer,
            "thinking": thinking,
            "sources": sources,
            "context": context[:500] + "..." if len(context) > 500 else context
        }
        
    except Exception as e:
        return {
            "answer": f"Error generating response: {str(e)}",
            "thinking": "Error in processing",
            "sources": [],
            "context": ""
        }

def generate_job_card_from_documents(llm, vector_store, query="Generate a comprehensive repair job card"):
    """Generate job card using document context"""
    # First retrieve relevant information
    analysis_result = analyze_with_rag(llm, query, vector_store)
    
    if not analysis_result["sources"]:
        return analysis_result
    
    # Generate structured job card
    job_card_prompt = f"""
    Based on the following document context, create a detailed repair job card:
    
    DOCUMENT CONTEXT:
    {analysis_result['context']}
    
    Create a comprehensive job card with:
    1. JOB CARD HEADER (Job ID, Date, Equipment, Priority)
    2. EQUIPMENT DETAILS (Make, Model, Serial Number if available)
    3. PROBLEM DESCRIPTION (Based on context)
    4. REQUIRED PARTS (Extract from context, include part numbers if available)
    5. REQUIRED TOOLS (Specific tools mentioned)
    6. STEP-BY-STEP PROCEDURE (Detailed steps from context)
    7. SAFETY PRECAUTIONS (All safety information)
    8. QUALITY CHECKS (Verification steps)
    9. ESTIMATED TIME (If mentioned in context)
    10. SKILL LEVEL REQUIRED
    11. SOURCES CITED (List all document sources used)
    
    IMPORTANT: Only include information found in the document context.
    If certain information is not available, mark as "Not specified in documents".
    """
    
    try:
        response = llm.invoke(job_card_prompt)
        
        # Store job card with metadata
        job_card_data = {
            "title": f"Job Card from {analysis_result['sources'][0]['source']}",
            "content": response.content,
            "sources": analysis_result["sources"],
            "timestamp": "Generated from documents",
            "query_used": query
        }
        
        return job_card_data
        
    except Exception as e:
        return {
            "title": "Error",
            "content": f"Error generating job card: {str(e)}",
            "sources": [],
            "timestamp": "Error",
            "query_used": query
        }

# UI Components
def display_document_info():
    """Display uploaded document information"""
    if st.session_state.uploaded_documents:
        st.sidebar.subheader("📚 Uploaded Documents")
        
        for doc_name in st.session_state.uploaded_documents:
            metadata = st.session_state.document_metadata.get(doc_name, {})
            
            with st.sidebar.expander(f"📄 {doc_name}", expanded=False):
                st.caption(f"Type: {metadata.get('type', 'unknown')}")
                st.caption(f"Status: {metadata.get('status', 'unknown')}")
                st.caption(f"Chunks: {metadata.get('chunks', 0)}")
                
                if "error" in metadata:
                    st.error(metadata["error"])

def display_thinking_process(thinking, answer, sources):
    """Display the AI's thinking process"""
    with st.expander("🤔 AI Thinking Process", expanded=True):
        st.markdown("### Step-by-Step Reasoning")
        st.markdown(thinking)
    
    st.markdown("### 📝 Answer")
    st.markdown(answer)
    
    if sources:
        with st.expander("🔍 Sources Used", expanded=False):
            for i, source in enumerate(sources):
                st.markdown(f"**Source {i+1}:** {source['source']}")
                st.caption(f"Page: {source['page']} | Relevance: {source['relevance']:.3f}")
                st.text(f"Preview: {source['content_preview']}")
                st.divider()

def main():
    st.set_page_config(
        page_title="Advanced Repair Job RAG System",
        page_icon="🔧",
        layout="wide"
    )
    
    init_session_state()
    
    # Title
    st.title("🔧 Advanced Repair Job RAG System")
    st.markdown("""
    Upload technical manuals and get **grounded, citation-backed responses** with AI reasoning.
    Prevents hallucinations by strictly using document context.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key
        api_key = st.text_input(
            "API Key",
            type="password",
            value=st.session_state.api_key,
            help="Enter your API key for AI services"
        )
        
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            if api_key:
                st.success("API key updated")
        
        st.divider()
        
        # Document Upload
        st.header("📁 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Upload technical manuals (PDF, TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Upload repair manuals, parts catalogs, technical documents"
        )
        
        if uploaded_files and st.session_state.api_key:
            process_button = st.button("Process Documents", type="primary")
            
            if process_button:
                with st.spinner("Processing documents..."):
                    # Clear previous state
                    st.session_state.uploaded_documents = []
                    st.session_state.document_metadata = {}
                    all_chunks = []
                    
                    # Process each file
                    for uploaded_file in uploaded_files:
                        filename = uploaded_file.name
                        
                        # Extract text
                        text, metadata, status = process_document(uploaded_file, filename)
                        
                        if status == "Success" and text:
                            # Create chunks
                            chunks = chunk_document(text, metadata)
                            
                            # Store metadata
                            st.session_state.uploaded_documents.append(filename)
                            st.session_state.document_metadata[filename] = {
                                "type": metadata[0]["type"] if metadata else "unknown",
                                "chunks": len(chunks),
                                "status": "Processed",
                                "size": len(text)
                            }
                            
                            all_chunks.extend(chunks)
                            
                            st.success(f"✅ {filename}: {len(chunks)} chunks")
                        else:
                            st.error(f"❌ {filename}: {status}")
                    
                    # Create vector store if we have chunks
                    if all_chunks and st.session_state.api_key:
                        with st.spinner("Creating search index..."):
                            embeddings = initialize_embeddings(st.session_state.api_key)
                            vector_store = create_vector_store(all_chunks, embeddings, st.session_state.api_key)
                            
                            if vector_store:
                                st.session_state.vector_store = vector_store
                                st.success(f"✅ Vector store created with {len(all_chunks)} chunks")
                            else:
                                st.error("Failed to create vector store")
        
        # Display document info
        display_document_info()
        
        # Quick actions
        if st.session_state.uploaded_documents:
            st.divider()
            st.header("⚡ Quick Actions")
            
            if st.button("🛠️ Generate Job Card from Documents"):
                st.session_state.generate_job_card = True
                st.rerun()
            
            if st.button("🧹 Clear All Documents"):
                st.session_state.uploaded_documents = []
                st.session_state.vector_store = None
                st.session_state.document_chunks = []
                st.session_state.document_metadata = {}
                st.rerun()
    
    # Main area
    tab1, tab2, tab3 = st.tabs(["💬 Document Q&A", "📋 Job Cards", "📊 Document Analysis"])
    
    # Tab 1: Document Q&A
    with tab1:
        st.header("Ask Questions About Your Documents")
        
        if not st.session_state.uploaded_documents:
            st.info("📤 Upload documents in the sidebar to ask questions about them.")
        elif not st.session_state.vector_store:
            st.warning("Documents uploaded but not indexed. Click 'Process Documents' in sidebar.")
        else:
            # Show document status
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Documents", len(st.session_state.uploaded_documents))
            with col2:
                total_chunks = sum(md.get("chunks", 0) for md in st.session_state.document_metadata.values())
                st.metric("Document Chunks", total_chunks)
            with col3:
                st.metric("Vector Store", "Ready" if st.session_state.vector_store else "Not Ready")
            
            # Query input
            query = st.text_area(
                "Ask a question about your technical documents:",
                placeholder="Example: What are the maintenance procedures for the hydraulic system? What parts are needed for pump repair? What safety precautions should be taken?",
                height=100
            )
            
            col1, col2 = st.columns([1, 3])
            with col1:
                ask_button = st.button("🔍 Analyze Documents", type="primary", use_container_width=True)
            
            # Process query
            if query and ask_button and st.session_state.api_key:
                with st.spinner("Analyzing documents..."):
                    # Initialize LLM
                    llm = initialize_llm(st.session_state.api_key)
                    
                    # Get RAG-based analysis
                    result = analyze_with_rag(llm, query, st.session_state.vector_store)
                    
                    # Display results
                    display_thinking_process(
                        result["thinking"],
                        result["answer"],
                        result["sources"]
                    )
                    
                    # Store in chat history
                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"Document query: {query}"
                    })
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                        "thinking": result["thinking"]
                    })
            
            # Chat history
            if st.session_state.messages:
                st.divider()
                st.subheader("💭 Conversation History")
                
                for i, message in enumerate(st.session_state.messages):
                    if message["role"] == "user":
                        with st.chat_message("user"):
                            st.markdown(message["content"])
                    else:
                        with st.chat_message("assistant"):
                            st.markdown(message["content"])
                            
                            # Show sources if available
                            if "sources" in message and message["sources"]:
                                with st.expander("View Sources", expanded=False):
                                    for source in message["sources"]:
                                        st.caption(f"{source['source']} (Page {source['page']})")
                
                if st.button("Clear Conversation"):
                    st.session_state.messages = []
                    st.rerun()
    
    # Tab 2: Job Cards
    with tab2:
        st.header("Generated Job Cards")
        
        # Generate job card from documents
        if "generate_job_card" in st.session_state and st.session_state.generate_job_card:
            if st.session_state.uploaded_documents and st.session_state.vector_store and st.session_state.api_key:
                with st.spinner("Generating job card from documents..."):
                    llm = initialize_llm(st.session_state.api_key)
                    
                    # Create job card
                    job_card_data = generate_job_card_from_documents(
                        llm,
                        st.session_state.vector_store,
                        "Extract all repair and maintenance information to create a comprehensive job card"
                    )
                    
                    if job_card_data:
                        st.session_state.repair_jobs.append(job_card_data)
                        st.success("Job card generated from documents!")
            
            # Reset the flag
            st.session_state.generate_job_card = False
        
        # Display job cards
        if not st.session_state.repair_jobs:
            st.info("No job cards generated yet. Use templates or generate from documents.")
        else:
            for i, job in enumerate(st.session_state.repair_jobs):
                with st.expander(f"🛠️ {job.get('title', 'Job Card')}", expanded=(i == 0)):
                    # Display job card content
                    st.markdown(job["content"])
                    
                    # Display sources if available
                    if job.get("sources"):
                        with st.expander("📚 Document Sources", expanded=False):
                            for source in job["sources"]:
                                st.caption(f"**{source['source']}** (Page {source['page']})")
                                st.text(f"Relevance: {source['relevance']:.3f}")
                    
                    # Action buttons
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.download_button(
                            "📥 Download",
                            job["content"],
                            file_name=f"job_card_{i+1}.md",
                            mime="text/markdown",
                            key=f"dl_{i}"
                        )
                    with col2:
                        if st.button("🔄 Regenerate", key=f"reg_{i}"):
                            # Placeholder for regeneration logic
                            st.info("Regeneration feature coming soon")
                    with col3:
                        if st.button("🗑️ Delete", key=f"del_{i}"):
                            st.session_state.repair_jobs.pop(i)
                            st.rerun()
    
    # Tab 3: Document Analysis
    with tab3:
        st.header("Document Analysis Dashboard")
        
        if not st.session_state.uploaded_documents:
            st.info("Upload documents to see analysis.")
        else:
            # Document statistics
            st.subheader("📊 Document Statistics")
            
            cols = st.columns(len(st.session_state.uploaded_documents))
            for idx, doc_name in enumerate(st.session_state.uploaded_documents):
                with cols[idx]:
                    metadata = st.session_state.document_metadata.get(doc_name, {})
                    st.metric(
                        doc_name[:20] + ("..." if len(doc_name) > 20 else ""),
                        f"{metadata.get('chunks', 0)} chunks"
                    )
            
            # Sample chunks
            st.subheader("🔍 Sample Document Chunks")
            if st.session_state.document_chunks:
                sample_chunks = st.session_state.document_chunks[:3]  # Show first 3 chunks
                
                for i, chunk in enumerate(sample_chunks):
                    with st.expander(f"Chunk {i+1}: {chunk.metadata.get('source', 'Unknown')} - Page {chunk.metadata.get('page', 1)}"):
                        st.text(chunk.page_content[:500] + ("..." if len(chunk.page_content) > 500 else ""))
            
            # Search test
            st.subheader("🔎 Test Document Search")
            test_query = st.text_input("Enter a test query to search documents:")
            
            if test_query and st.session_state.vector_store:
                test_results = retrieve_relevant_chunks(test_query, st.session_state.vector_store, k=3)
                
                if test_results:
                    st.success(f"Found {len(test_results)} relevant chunks")
                    
                    for result in test_results:
                        with st.expander(f"📄 {result['source']} (Score: {result['score']:.3f})"):
                            st.caption(f"Page: {result['page']}")
                            st.text(result['content'][:300] + "...")
    
    # Footer
    st.markdown("---")
    st.caption("""
    **Advanced RAG System Features:**
    - ✅ Document-based responses with citations
    - ✅ AI reasoning process visualization
    - ✅ Hallucination prevention through source grounding
    - ✅ Vector search for precise information retrieval
    - ✅ Job card generation from technical documents
    """)

if __name__ == "__main__":
    main()