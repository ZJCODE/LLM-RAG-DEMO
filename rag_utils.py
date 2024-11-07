import os
import time
import pathlib
import uuid
import hashlib
from tqdm import tqdm
from typing import Generator
import pymupdf4llm
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import chromadb.utils.embedding_functions as embedding_functions

class DocLoader:
    """
    A class to load and process documents from a specified directory.
    
    Attributes:
        path (str): The path to the directory containing the documents.
        chunk_size (int): The size of the text chunks to split the documents into.
        chunk_overlap (int): The amount of overlap between text chunks.
        enable_logging (bool): Flag to show progress of loading documents.
        docs (Generator): A generator yielding processed markdown documents.
    """
    
    def __init__(self, path: str,chunk_size=512,chunk_overlap=128,enable_logging=True):
        """
        Initializes the DocLoader with the specified path and loads the documents.
        
        Args:
            path (str): The path to the directory containing the documents.
            chunk_size (int): The size of the text chunks to split the documents into.
            chunk_overlap (int): The amount of overlap between text chunks.
            enable_logging (bool): Flag to show progress of loading documents.
        """
        self.path = path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_logging = enable_logging
        if self.enable_logging:
            print("\nLoading files from {}".format(self.path))
        self.docs = self.load()
        self.file_count = self.count_total_markdown_files()

    def load(self) -> Generator:
        """
        Loads the documents by processing PDFs and loading markdown files.
        
        Returns:
            Generator: A generator yielding processed markdown documents.
        """
        self.process_pdfs()
        return self.load_markdown_files()

    def process_pdfs(self):
        """
        Converts PDF files in the specified directory and all subdirectories to markdown format.
        """
        if self.enable_logging:
            print("\nProcessing PDFs")
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".pdf"):
                    base_name = os.path.splitext(file)[0]
                    md_file_path = os.path.join(root, f"{base_name}_pdf_converted.md")
                    if not os.path.exists(md_file_path):
                        if self.enable_logging:
                            print(f"Converted {os.path.join(root, file)} to markdown")
                        md_text = pymupdf4llm.to_markdown(os.path.join(root, file))
                        pathlib.Path(md_file_path).write_bytes(md_text.encode())
                    else:
                        if self.enable_logging:
                            print(f"Skipping {os.path.join(root, file)} as it has already been converted to markdown")

    def count_total_markdown_files(self) -> int:
        """
        Counts the total number of markdown files in the specified directory and all subdirectories.
        
        Returns:
            int: The total number of markdown files.
        """
        count = 0
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".md"):
                    count += 1
        return count
    
    def load_markdown_files(self) -> Generator:
        """
        Loads markdown files from the specified directory and all subdirectories.
        
        Yields:
            tuple: A tuple containing header splits and text chunks of the markdown file.
        """
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".md"):
                    yield self.load_markdown(os.path.join(root, file), self.chunk_size, self.chunk_overlap)

    @staticmethod
    def load_markdown(file: str, chunks=512, chunk_overlap=128) -> tuple:
        """
        Loads and splits a markdown file into headers and text chunks.
        
        Args:
            file (str): The path to the markdown file.
            chunks (int): The size of the text chunks to split the documents into.
            chunk_overlap (int): The amount of overlap between text chunks.
        
        Returns:
            tuple: A tuple containing the file name, text chunks, and metadata.
        """
        with open(file, 'r') as f:
            md_text = f.read()

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        md_header_splits = markdown_splitter.split_text(md_text)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunks, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(md_header_splits)
        
        documents = [chunk.page_content for chunk in chunks]
        for chunk in chunks:
            chunk.metadata.update({"file": file})
        metadatas = [chunk.metadata for chunk in chunks]
        
        return file,documents,metadatas

    @staticmethod
    def load_single_file(file: str, chunks=512, chunk_overlap=128) -> tuple:
        # file can be a pdf or a markdown file
        # if file is a pdf, convert it to markdown then call load_markdown
        if file.endswith(".pdf"):
            base_name = os.path.splitext(file)[0]
            md_file_path = f"{base_name}_pdf_converted.md"
            if not os.path.exists(md_file_path):
                md_text = pymupdf4llm.to_markdown(file)
                pathlib.Path(md_file_path).write_bytes(md_text.encode())
            return DocLoader.load_markdown(md_file_path, chunks, chunk_overlap)
        elif file.endswith(".md"):
            return DocLoader.load_markdown(file, chunks, chunk_overlap)
        else:
            raise ValueError("File must be a PDF or markdown file")
class StoreIndex:
    """
    StoreIndex is a class for managing a collection of documents with optional embedding functions.
    
    Attributes:
        db_path (str): Path to the database.
        collection_name (str): Name of the collection.
        extra_exembedding (bool): Flag to use extra embedding functions.
        hash_text_uuid (bool): Flag to hash text to UUID.
        enable_logging (bool): Flag to show progress of adding documents.
        client (chromadb.PersistentClient): Persistent client for the database.
        collection (chromadb.Collection): Collection of documents.
    """
    
    def __init__(self, db_path, collection_name, extra_exembedding=True, hash_text_uuid=True,enable_logging=True):
        """
        Initialize the StoreIndex with the given parameters.
        
        Args:
            db_path (str): Path to the database.
            collection_name (str): Name of the collection.
            extra_exembedding (bool): Flag to use extra embedding functions.
            hash_text_uuid (bool): Flag to hash text to UUID.
        """
        load_dotenv()
        self.db_path = db_path
        self.collection_name = collection_name
        self.extra_exembedding = extra_exembedding
        self.hash_text_uuid = hash_text_uuid
        self.enable_logging = enable_logging
        if not os.path.exists(self.db_path):
            print(f"\nCreating database at {self.db_path}.")
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self._create_collection()
        if self.enable_logging:
            print(f"\nConnected to collection {self.collection_name} in database {self.db_path} for indexing.")
        
    def _create_collection(self):
        """
        Create or get the collection with optional embedding functions.
        
        Returns:
            chromadb.Collection: The created or retrieved collection.
        """
        if self.extra_exembedding:
            
            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_BASE_URL")
            embedding_model = os.environ.get("OPENAI_EMBEDDING_NAME", "text-embedding-3-small")
            
            if not api_key:
                raise ValueError("API key must be set in environment variables.")
            
            if base_url:
                openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=api_key,
                    api_base=base_url,
                    model_name=embedding_model
                )
            else:
                openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=api_key,
                    model_name=embedding_model
                )
            if not self.exists(self.db_path, self.collection_name):
                print("\nCreating collection {} in the database {}.".format(self.collection_name, self.db_path))
            return self.client.get_or_create_collection(name=self.collection_name, embedding_function=openai_ef)
        return self.client.get_or_create_collection(name=self.collection_name)

    def add(self, documents, metadatas=None):
        """
        Add documents and their corresponding metadata to the collection.
        
        Args:
            documents (Union[str, List[str]]): Documents to be added. Can be a single document or a list of documents.
            metadatas (Union[Mapping[str, Union[str, int, float, bool]], List[Mapping[str, Union[str, int, float, bool]]]]): 
                Metadata for the documents. Can be a single mapping or a list of mappings.
            enable_logging (bool): Flag to show progress of adding documents.
        
        Raises:
            ValueError: If the lengths of metadatas and documents do not match.
        """
        documents = documents if isinstance(documents, list) else [documents]
        
        if self.hash_text_uuid:
            ids = [self._generate_sha256_hash_from_text(doc) for doc in documents]
        else:
            ids = [f"{uuid.uuid4()}" for _ in range(len(documents))]
        
        ids = list(set(ids))
        
        existing_ids = set(self.collection.get(ids=ids)["ids"])
        
        if len(existing_ids) == len(ids):
            if self.enable_logging:
                print("All documents already exist in the collection.")
            return
        if self.enable_logging:
            print(f"Totale {len(documents)} | Adding {len(documents) - len(existing_ids)} new documents to the collection.")
        
        if metadatas:
            metadatas = metadatas if isinstance(metadatas, list) else [metadatas]    
            if len(metadatas) != len(documents):
                raise ValueError("metadatas and documents should have the same length")
            filtered_documents, filtered_metadatas, filtered_ids = [], [], []
            for doc, meta, id in zip(documents, metadatas, ids):
                if id not in existing_ids:
                    filtered_documents.append(doc)
                    filtered_metadatas.append(meta)
                    filtered_ids.append(id)
            self.collection.add(documents=filtered_documents, ids=filtered_ids, metadatas=filtered_metadatas)
        else:
            filtered_documents, filtered_ids = [], []
            for doc, id in zip(documents, ids):
                if id not in existing_ids:
                    filtered_documents.append(doc)
                    filtered_ids.append(id)
            self.collection.add(documents=filtered_documents, ids=filtered_ids)

    def index_from_doc_loader(self, doc_loader: DocLoader):
        """
        Index documents from a DocLoader object.
        
        Args:
            doc_loader (DocLoader): DocLoader object to load documents from.
        """        
        
        for file, documents, metadatas in tqdm(doc_loader.docs,
                                               total= doc_loader.file_count, 
                                               desc="Indexing Progress"):
            if self.enable_logging:
                print(f"\nIndexing documents from {file}.")
            self.add(documents=documents, metadatas=metadatas)
        if self.enable_logging:
            print("\nIndexing completed, total documents indexed: {}.".format(doc_loader.file_count))

    def peek(self, n=10):
        """
        Get the first n documents in the collection.
        
        Args:
            n (int): Number of documents to return.
        
        Returns:
            List[str]: The first n documents in the collection.
        """
        return self.collection.peek(n)
    
    def clear(self):
        """
        Clear the collection and create a new one.
        """
        print(f"Deleting collection {self.collection_name} in the database {self.db_path}.")
        self.client.delete_collection(name=self.collection_name)
        self.collection = self._create_collection()

    def delete(self):
        """
        Delete the collection.
        """
        print(f"Deleting collection {self.collection_name} in the database {self.db_path}.")
        self.client.delete_collection(name=self.collection_name)

    @staticmethod
    def exists(db_path: str, collection_name: str) -> bool:
        """
        Check if the db and collection exists.
        
        Args:
            db_path (str): Path to the database.
            collection_name (str): Name of the collection.
        
        Returns:
            bool: True if the collection exists, False otherwise.
        """
        
        if not os.path.exists(db_path):
            return False
        else:
            client = chromadb.PersistentClient(path=db_path)
            try:
                client.get_collection(name=collection_name)
                return True
            except:
                return False
            
    @staticmethod
    def split_list(lst, n):
        # Calculate the size of each part
        k, m = divmod(len(lst), n)
        return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

    @staticmethod
    def _generate_sha256_hash_from_text(text: str) -> str:
        """
        Generate a SHA-256 hash from the given text.
        
        Args:
            text (str): The text to hash.
        
        Returns:
            str: The SHA-256 hash of the text.
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()


class SearchEngine:
    """
    A search engine that uses a language model to extract key words from a query and search a database.

    Attributes:
        extra_exembedding (bool): Flag to use extra embedding functions.
        llm (OpenAI): The language model instance.
        llm_model (str): The name of the language model.
    """
    def __init__(self, db_path='db', collection_name='demo',extra_exembedding=True,enable_logging=True):
        """
        Initializes the SearchEngine with a database path and collection name.

        Args:
            db_path (str): The path to the database.
            collection_name (str): The name of the collection in the database.
            extra_exembedding (bool): Flag to use extra embedding functions.
        """
        
        if not os.path.exists(db_path):
            raise ValueError(f"Database path '{db_path}' does not exist.")

        self.extra_exembedding = extra_exembedding
        self.enable_logging = enable_logging

        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        embedding_model = os.environ.get("OPENAI_EMBEDDING_NAME", "text-embedding-3-small")
        
        if not api_key:
            raise ValueError("API key must be set in environment variables.")
        
        client = chromadb.PersistentClient(path=db_path)
        
        try:
            if self.extra_exembedding:
                if base_url:
                    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                        api_key=api_key,
                        api_base=base_url,
                        model_name=embedding_model
                    )
                else:
                    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                        api_key=api_key,
                        model_name=embedding_model
                    )
            
                self.collection = client.get_collection(name=collection_name,embedding_function=openai_ef)
            else:
                self.collection = client.get_collection(name=collection_name)
            if self.enable_logging:
                print(f"\nConnected to collection {collection_name} in database {db_path} for searching.")
        except:
            raise ValueError(f"Collection '{collection_name}' does not exist in database at path '{db_path}'.")
        
        self.llm = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.llm_model = os.environ.get("OPENAI_MODEL_NAME","gpt-4o-mini")

    def search(self, query, n_results=1):
        """
        Searches the database using a query.

        Args:
            query (str): The search query.
            n_results (int): The number of results to return.

        Returns:
            list: The search results.
        """

        return self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        
    
    def hybrid_search(self, query, text_results=3,semantic_results=3):
        """
        Performs a hybrid search using both text and semantic recall.

        Args:
            query (str): The search query.
            text_results (int): The number of text-based results to return.
            semantic_results (int): The number of semantic-based results to return.

        Returns:
            list: The combined search results.
        """
        if text_results + semantic_results < 1:
            raise ValueError("The sum of text_results and semantic_results must be greater than 0.")
        
        text_recall = None
        text_semanti_recall = None
        
        if text_results > 0:
            simple_key_word = self._extract_simple_key_word_from_query(query)
            
            if self.enable_logging:
                print(f"Simple key word for text recall: {simple_key_word}")
            
            text_recall = self.collection.query(
                    query_texts=[query],
                    n_results=text_results,
                    where_document={"$contains": simple_key_word}
                )
        
        if semantic_results > 0:
            key_words = self._extract_key_words_from_query(query)
            
            if self.enable_logging:
                print(f"Key words for semantic recall: {key_words}")
            
            text_semanti_recall = self.collection.query(
                query_texts=[query],
                n_results=semantic_results,
                where_document={"$or": [{"$contains": kw} for kw in key_words]}
                )

        simple_search_recall = self.search(query, n_results = text_results + semantic_results)
        
        results = self.combine_objects(text_recall,text_semanti_recall)
        
        results = self.combine_objects(results,simple_search_recall)
        
        return results
    
    def _extract_simple_key_word_from_query(self, query):
        """
        Extracts key words from the query using the language model to do a simple split.

        Args:
            query (str): The search query.

        Returns:
            list: A list of key words.
        """
        prompt = (
            f"Analyze the query below, identify a single key search term for recal.\n\n"
            f"{query}\n\n"
            f"Just return the final key word."
        )
        
        try:
            response = self.llm.chat.completions.create(
                messages=[{"role": "user","content": prompt}],
                model=self.llm_model,
                temperature=0.0,
                max_tokens=1024,
            )
            key_word = response.choices[0].message.content.strip()
            return key_word
        
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return None
    
    def _extract_key_words_from_query(self, query):
        """
        Extracts key words from the query using the language model.

        Args:
            query (str): The search query.

        Returns:
            list: A list of key words.
        """
        prompt = (
            f"Analyze the query below, identify key search terms for recall, and expand the query using synonyms, lemmatization and ensure to include both single words and phrases.\n\n"
            f"{query}\n\n"
            f"Just return the final key words in a comma-separated list like 'key_word1,key_word2,key_word3',at most five key words."
        )
        
        try:
            response = self.llm.chat.completions.create(
                messages=[{"role": "user","content": prompt}],
                model=self.llm_model,
                temperature=0.1,
                max_tokens=1024,
            )
            key_words = [kw.strip() for kw in response.choices[0].message.content.split(",")]
            
            prompt_translation = (
                f"Translate the key words below into both Chinese and English.\n\n"
                f"{key_words}\n\n"
                f"Just return the final translated key words in a comma-separated list like 'key_word1,key_word2,key_word3,关键词1,关键词2,关键词3'."
            )
            
            response = self.llm.chat.completions.create(
                messages=[{"role": "user","content": prompt_translation}],
                model=self.llm_model,
                temperature=0.1,
                max_tokens=1024,
            )
            key_words = [kw.strip() for kw in response.choices[0].message.content.split(",")]
            
            return key_words
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return []
        
    @staticmethod
    def combine_objects(obj1, obj2):
        """
        Combines two search result objects.

        Args:
            obj1 (dict): The first search result object.
            obj2 (dict): The second search result object.

        Returns:
            dict: The combined search result object.
        """
        if obj1 is None:
            return obj2
        if obj2 is None:
            return obj1
        
        obj1_ids = set(obj1['ids'][0])
        
        result = {key: obj1[key] for key in ['ids', 'distances', 'metadatas', 'documents']}
        
        for index, obj_id in enumerate(obj2['ids'][0]):
            if obj_id not in obj1_ids:
                for key in result:
                    result[key][0].append(obj2[key][0][index])
                
        return result

        
class RAGChatbot:
    """
    A chatbot that uses a Retrieval-Augmented Generation (RAG) approach to answer questions.
    
    Attributes:
        se (SearchEngine): The search engine used to retrieve documents.
        enable_logging (bool): Flag to show progress messages.
        llm (OpenAI): The language model instance.
        llm_model (str): The name of the language model.
    """
    
    def __init__(self, search_engine: SearchEngine,hybrid_search = False, enable_logging=True):
        """
        Initializes the RAGChatbot with a search engine and optional progress display.
        
        Args:
            search_engine (SearchEngine): The search engine to use for document retrieval.
            enable_logging (bool): Whether to show progress messages. Default is True.
        """
        self.se = search_engine
        self.enable_logging = enable_logging
        self.hybrid_search = hybrid_search
        self.llm, self.llm_model = self.init_llm()
        if self.enable_logging:
            print("\nConnected to Search Engine and Language Model")
        self.messages = []
        
    def clear_history(self):
        """
        Clears the chat history.
        """
        self.messages = []
        
    def init_llm(self):
        """
        Initializes the language model using environment variables for API key and base URL.
        
        Returns:
            tuple: A tuple containing the language model instance and the model name.
        
        Raises:
            ValueError: If the API key is not set in environment variables.
        """
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        
        if not api_key:
            raise ValueError("API key must be set in environment variables.")
        
        llm = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        
        llm_model = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")
        
        return llm, llm_model

    def answer(self, question, reference_num=2,temperature = 0.1,max_tokens = 1024, history_length = 6):
        """
        Answers a question using the RAG approach by retrieving documents and generating a response.
        
        Args:
            question (str): The question to answer.
            reference_num (int): The number of documents to retrieve. Default is 2.
            temperature (float): The sampling temperature for the language model. Default is 0.1.
            max_tokens (int): The maximum number of tokens to generate. Default is 1024.
        
        Returns:
            tuple: A tuple containing the generated answer and the retrieved documents.
        """
        start_search_time = time.time()
        if self.hybrid_search:
            result = self.se.hybrid_search(question, text_results=reference_num,semantic_results=reference_num)
        else:
            result = self.se.search(question, n_results=reference_num)
        documents = result['documents'][0]
        documents_path = [x['file'] for x in  result['metadatas'][0]]
        documents_res = [{'document':document,'path':document_path} for document,document_path in zip(documents,documents_path)]
        end_search_time = time.time()
        search_duration = end_search_time - start_search_time
    
        if self.enable_logging:
            print(f"\nSearch completed in {search_duration:.2f} seconds")
    
        prompt = (
            f"Context information is below.\n\n"
            f"----------------------------------------\n\n"
            f"{documents_res}\n\n"
            f"----------------------------------------\n\n"
            f"Given the context information and prior chat, answer the question below.\n\n"
            f"{question}\n"
        )
        
        if history_length:
            messages = self.messages[-history_length:] + [{"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]
        
        start_llm_time = time.time()
        response = self.llm.chat.completions.create(
            messages=messages,
            model=self.llm_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.messages.append({"role": "user", "content": question})
        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        
        end_llm_time = time.time()
        llm_duration = end_llm_time - start_llm_time
        
        if self.enable_logging:
            print(f"\nLLM completed in {llm_duration:.2f} seconds")
        
        return response.choices[0].message.content, documents_res