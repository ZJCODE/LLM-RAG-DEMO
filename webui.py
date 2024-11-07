import streamlit as st

from rag_utils import DocLoader,StoreIndex,SearchEngine,RAGChatbot

# st.set_page_config(layout="wide")

db_path,collection_name = 'db','demo'

# StoreIndex(db_path=db_path,collection_name=collection_name,enable_logging=False)

if "SearchEngine" not in st.session_state:
    st.session_state.SearchEngine = SearchEngine(db_path=db_path, collection_name=collection_name)

if "RAGChatbot" not in st.session_state:
    st.session_state.RAGChatbot = RAGChatbot(st.session_state.SearchEngine,hybrid_search=True)

if 'StoreIndex' not in st.session_state:
    st.session_state.StoreIndex = StoreIndex(db_path=db_path,collection_name=collection_name)

with st.sidebar:
    
    # st.write("### Current Database Info")
    
    # st.markdown(f"- Database Path: {db_path}")
    # st.markdown(f"- Collection Name: {collection_name}")
    
    # st.markdown("---")
    
    st.write("### Parameters")
    reference_num = st.number_input("reference_num", min_value=1, max_value=20, value=3)
    max_tokens = st.number_input("max_tokens", min_value=1, max_value=8000, value=1000)

    st.write("### Add New Document")
    
    uploaded_file = st.file_uploader(label = "upload file", type=("pdf", "md"),label_visibility='collapsed')

    # save uploaded file to temp folder
    if uploaded_file:
        file_path = f"temp/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.StoreIndex.add(*DocLoader.load_single_file(file_path)[1:]) # add extra single doc if needed

    st.write("### Load Documents From Folder")
    folder_path = st.text_input("Absolute Folder Path")
    if st.button("Load"):
        if folder_path:
            dl = DocLoader(path=folder_path)
            st.session_state.StoreIndex.index_from_doc_loader(dl)    
        
    st.write("### Display Control")
    show_refs = st.checkbox("Show References", value=True)
    

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input():
    
    if prompt == '/c':
        st.session_state.messages = []
        st.session_state.RAGChatbot.clear_history()
        st.rerun()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.spinner('Searching & Thinking...'):
        res,doc = st.session_state.RAGChatbot.answer(prompt,reference_num=reference_num,temperature=0,max_tokens=max_tokens)
    msg = res 
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
    
    refs = "参考文档 :\n\n" +  "\n".join(set(["- " + x['path'] for x in doc]))
    if show_refs:
        st.session_state.messages.append({"role": "system", "content": refs})
        st.chat_message("system").write(refs)
    