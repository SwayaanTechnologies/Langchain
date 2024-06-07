# **Langchain**

These models are trained on massive amounts of text data to learn patterns and entity relationships in the language. LLMs use deep learning algorithms to process and understand natural language, performing tasks such as translating, analyzing sentiments, and generating new text.

![logo](https://media.gettyimages.com/id/1801115823/photo/in-this-photo-illustration-the-langchain-logo-is-displayed.jpg?b=1&s=594x594&w=0&k=20&c=OpkcRRc6G8I_-jYYk4Tgu5gWVtgYilTypQ4naXcNJqU=)

## **TABLE OF CONTENT**

**1.** [**Introduction to Langchain**](#Introduction-to-Langchain)

**2.** [**Components**](#Components)

* [**Schema**](#Schema)
* [**Models**](#Models)
* [**Prompts**](#Prompts)
* [**Parsers**](#Parsers)
* [**Indexes**](#Indexes)

  - [**Document Loading**](#Document-Loading)   

  -  [**Documnet Splitting**](#Documnet-Splitting)  

  - [**Vectors and Embeddings**](#Vectors-and-Embeddings) 

  - [**Retrieval**](#Retrieval)

* [**Memory**](#Memory)

  - [**Chat Message History**](#Chat-Message-History)

  - [**Conversation Buffer Memory**](#Conversation-Buffer-Memory)

  - [**Conversation Buffer Window Memory**](#Conversation-Buffer-Window-Memory)

  - [**Conversation Token Buffer Memory**](#Conversation-Token-Buffer-Memory)

  - [**Conversation Summary Memory**](#Conversation-Summary-Memory)

  - [**Knowledge Graph Memory**](#Knowledge-Graph-Memory)

  - [**Entity Memory**](#Entity-Memory)

* [**Chains**](#Chains)
* [**Agents**](#Agents)

**3.** [**Embeddings**](#Embeddings)

**4.** [**Chain of Thought**](#Chain-of-Thought)

**5.** [**Retriever Augmented Generator**](#Retriever-Augmented-Generator)

**6.** [**Advanced Retrieval Techniques**](#Advanced-Retrieval-Techniques)

**7.** [**Transformers**](#Transformers)

**8.** [**References**](#References)

---

## **Introduction to Langchain**

* Langchain is an open-source framework that equips developers with the necessary tools to create applications powered by large language models (LLMs). Langchain is similar to Lego blocks for LLMs; you can use multiple models for various behaviors for different tasks without having to learn each one from scratch, and then you can create pipelines using Langchain speed un application development.


  1. [**Evolve**](#Evolve)
  2. [**Why do we need Langchain?**](#Why-do-we-need-Langchain?)

---

### **Evolve**

* The journey of LangChain began as an ambitious project to overcome the limitations of early language models. Its evolution is marked by significant milestones that reflect the rapid advancement of AI and NLP technologies. Initially, language models were constrained by simplistic rule-based systems that lacked the ability to understand context or generate natural-sounding text. As machine learning and deep learning techniques matured, the foundation for LangChain was set.

* The advancements in transfer learning further propelled LangChain, making it possible to fine-tune models on specific datasets. This adaptability made LangChain a versatile tool for developers in various fields.

* The integration of modular components for specialized linguistic tasks expanded LangChain’s capabilities. Developers could extend LangChain’s functionality by adding or removing modules tailored to their needs, such as sentiment analysis, language translation, and more.

* Throughout its history, LangChain has placed a significant focus on context retention. Early language models struggled to maintain context over extended conversations, but LangChain introduced advanced memory mechanisms, allowing it to remember and reference past interactions, thereby creating more natural and engaging dialogues.

* Today, LangChain stands as a testament to the progress in AI conversational systems. With each update and refinement, it has become more sophisticated, more intuitive, and more capable of delivering experiences that closely mimic human interaction. It’s a story of continual improvement and innovation, with the promise of further advancements as the AI field evolves.

* LangChain’s ongoing development is driven by a community of researchers, developers, and enthusiasts who are relentlessly pushing the boundaries of what’s possible in AI. As we look back at its brief but impactful history, it is clear that LangChain is not just following the trends in AI development—it is setting them, paving the way for a future where conversational AI becomes an integral part of our daily lives. It’s exciting to think about what the future holds for LangChain and AI in general!

---

### **Why do we need Langchain?**

* Most of the LLMs(OpenAi, Al 21 Labs, LLaMA...) are not up to date

* They are not good at Domain Knowledge and fail when working with Proprietary data

* Working with different LLMs may become a tedious task

* **Autonomous agents:** LangChain can be used to create autonomous agents that can write code, run tests, and deploy applications using natural language commands. This is particularly useful for automating repetitive tasks and improving productivity.

* **Agent simulations:** LangChain can be used to simulate the behavior and interactions of multiple agents in a sandbox environment. This can be used to test the long-term memory and social skills of language models or explore how they react to different events or scenarios.

* **Personal assistants:** LangChain can be used to create personal assistants that can access and manipulate user data, remember user preferences and history, and perform various tasks such as booking flights, ordering food, or sending emails. This can greatly enhance the user experience and make interactions with technology more natural and intuitive.

* **Question answering:** LangChain can be used to create question-answering applications that can extract relevant information from text, images, audio, or video files, and provide concise and accurate answers to user queries. This can be particularly useful in fields like customer service, where quick and accurate responses are crucial.

---

## **Components**

1. [**Schema**](#Schema)

2. [**Models**](#Models)

3. [**Prompts**](#Prompts)

4. [**Parsers**](#Parsers)

5. [**Indexes**](#Indexes)

    * [**Document Loading**](#Document-Loading)

    * [**Documnet Splitting**](#Documnet-Splitting)

    * [**Vectors and Embeddings**](#Vectors-and-Embeddings)

    * [**Retrevial**](#retrevial)

6. [**Memory**](#Memory)

    * [**Chat Message History**](#Chat-Message-History)

    * [**Conversation Buffer Memory**](#Conversation-Buffer-Memory)

    * [**Conversation Buffer Window Memory**](#Conversation-Buffer-Window-Memory)

    * [**Conversation Token Buffer Memory**](#Conversation-Token-Buffer-Memory)

    * [**Conversation Summary Memory**](#Conversation-Summary-Memory)

    * [**Knowledge Graph Memory**](#Knowledge-Graph-Memory)

    * [**Entity Memory**](#Entity-Memory)

7. [**Chains**](#Chains)

8. [**Agents**](#Agents)

9. [**Embeddings**](#Embeddings)

![Components](img/companes.png)

---

### **Schema**

* The schema in LangChain can be defined using various techniques and languages, depending on the specific requirements and technologies used in the project. Commonly used schema definition languages include SQL (Structured Query Language), JSON (JavaScript Object Notation), and YAML (YAML Ain’t Markup Language).

* By defining a clear and consistent schema, LangChain ensures that data is organized and standardized, facilitating efficient data retrieval and manipulation. This is crucial for the performance and reliability of applications built with LangChain. It also ensures compatibility and interoperability between different components, making it easier for developers to build and manage their applications.

**EXAMPLE**

```python
from secret_key import hugging_facehub_key
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "<your-api-key>"
```

1. **Import Libraries and Set Environment:**

* **import os:** Operating system module for setting environment variables.

* **from secret_key import hugging_facehub_key:** Import the API token for authentication.

* Set the Hugging Face API token in the environment.

```python
#SCHEMA

from langchain.llms import HuggingFaceHub
from langchain.schema import HumanMessage, SystemMessage

# Initialize Hugging Face model from the Hub
chat = HuggingFaceHub(
    repo_id="google/flan-t5-large", 
    model_kwargs={"temperature": 0.7, "max_length": 512}
)

# Create messages
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Translate this sentence from English to Nepali. I love programming.")
]
prompt = "\n".join([msg.content for msg in messages])

# Generate response for the single prompt
response = chat.predict(prompt)
print(response)

# Multiple sets of messages
batch_messages = [
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="I love programming.")
    ],
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="I love artificial intelligence.")
    ],
]

# Convert each set of messages to a single string prompt
prompts = ["\n".join([msg.content for msg in messages]) for messages in batch_messages]

# Get completions by passing in the list of string prompts
responses = [chat.predict(prompt) for prompt in prompts]

# Print the responses
for response in responses:
    print(response)
```

2. **Initialize Hugging Face Model:**

  * **from langchain.llms import HuggingFaceHub:** Import the HuggingFaceHub class.
  * Initialize the HuggingFaceHub model with specified parameters, such as the model ID and model parameters.

3. **Creating Messages:**
   
  * **SystemMessage(content="..."):** Create system messages with specified content.

  * **HumanMessage(content="..."):** Create user messages with specified content.

  * Concatenate system and user messages into a single prompt.

4.  **Generating Responses for Single Prompt:**

  * Use the `predict` method of the `chat` object to generate a response for the single prompt.

  * Print the generated response.

5. **Multiple Sets of Messages:**

  * Define multiple sets of system and user messages in `batch_messages`.

  * Convert each set of messages into a single prompt by concatenating them.

  * Store all prompts in the `prompts` list.

6. **Generate Responses for Batch Prompts:**

  * Use a list comprehension to generate responses for each prompt in the `prompts` list.

  * Store all generated responses in the `responses` list.

7. **Print Responses:**

  * Iterate over the responses list and print each generated response.

### **Models**

* models, such as GPT-4, are trained on vast amounts of text data and can generate human-like text based on the input they are given. They are the core of LangChain applications, enabling capabilities like natural language understanding and generation.

* LangChain provides a standard interface for interacting with these models, abstracting away many of the complexities involved in working directly with LLMs. This makes it easier for developers to build applications that leverage the power of these models.

![Models](img/model.jpg)

**EXAMPLE**

1. **Loading Environment Variable**

```python
from secret_key import hugging_facehub_key
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hugging_facehub_key
```

* **Purpose:** This block imports your API key for Hugging Face from a separate file (`secret_key.py`) and sets it as an environment variable. This key is required to authenticate requests to the Hugging Face API.

2. **Handling Model Deprecation**

```python
# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Set the model variable based on the current date
if current_date < datetime.date(2024,6,12):
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

print (llm_model)
```

* **Purpose:** This block checks the current date and sets the model name based on whether the current date is before or after June 12, 2024. This is to handle the deprecation of a specific model.

**Initializing the Language Model**

```python
from langchain import HuggingFaceHub
# Define LLM
repo_id = "google/flan-t5-large"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options

llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64}
)
```

* **Purpose:** This block initializes a language model (LLM) from the Hugging Face Hub using the `langchain` library. Here, the model being used is `google/flan-t5-large`. The `model_kwargs` parameter is used to set the temperature (which controls the randomness of the output) and the maximum length of the generated text.

---

### **Prompts**

A language model prompt is a user-provided set of instructions or input designed to guide the model's response. This aids the model in understanding the context and producing relevant output, whether it involves answering questions, completing sentences, or participating in a conversation

**EXAMPLE**

````python
# Prompts
customer_email = """
Arrr, I be fuming that me blender lid 
flew off and splattered me kitchen walls 
with smoothie! And to make matters worse, 
the warranty don't cover the cost of 
cleaning up me kitchen. I need yer help 
right now, matey!
"""

# Define the style
style = "American English (Formal Business Tone)"
# Create the full prompt
prompt = f"""Translate the following text into a style that is {style}:

```{customer_email}```
"""
````

* **Purpose:** This block defines the text to be translated (`customer_email`) and the desired style of the translation (`American English (Formal Business Tone)`). It then constructs a prompt by embedding the email text within a request for translation. Finally, it sends this prompt to the language model and prints the response.

---

### **Parsers**

Output parsers are responsible for taking the output of an LLM and transforming it to a more suitable format. This is very useful when you are using LLMs to generate any form of structured data.

**EXAMPLE**

```python
# Output Parsers
print(prompt)
response = llm(prompt)
print(response)
```

Here, the prompt string is printed, and the `llm` function is called with the prompt as input. The generated completion is then printed.

**Hugging Face Integration**

````python
# Hugging Face Integration
from transformers import pipeline

def huggingface_completion(prompt, model_name="text-generation", max_length=100, **kwargs):
    # Load Hugging Face pipeline with max_length parameter
    huggingface_pipeline = pipeline(model_name, max_length=max_length, **kwargs)
    # Generate completion
    return huggingface_pipeline(prompt)

# Example usage
completion = huggingface_completion("Translate the text that is delimited by triple backticks into a style that is American English in a calm and respectful tone. text: ```Arrr, I be fuming that me blender lid flew off and splattered me kitchen walls with smoothie! And to make matters worse, the warranty don't cover the cost of cleaning up me kitchen. I need yer help right now, matey!```")

print(completion)
````

This part integrates with the Hugging Face pipeline for text generation. It defines a function huggingface_completion that takes a prompt and optional parameters, and returns a completion generated by the specified model. Finally, it demonstrates the usage of this function by generating a completion for a specific prompt and printing it.

---

### **Indexes**

* Indexes in the context of language models (LLMs) refer to structured representations of documents that facilitate efficient interaction with the documents. These indexes play a crucial role in the retrieval of relevant documents in response to user queries.

* Imagine you have a vast collection of text documents. Without indexes, searching through this collection for relevant information would be like looking for a needle in a haystack. Indexes help organize this information in a way that makes it easier for LLMs to quickly find and retrieve the most relevant documents based on a user's query.

* The primary use of indexes in chains is in the retrieval step. This involves taking a user's query and using the index to identify and return the most relevant documents. However, indexes can be used for other purposes besides retrieval, and retrieval itself can employ other methods besides indexes to find relevant documents.

* It's important to note that indexes are typically used for unstructured data, such as text documents. For structured data like SQL tables or APIs, different methods are employed.

* LangChain primarily supports indexes and retrieval mechanisms centered around vector databases. These databases store documents as vectors, which enables efficient searching and retrieval based on similarities between vectors.

  * **Document Loading:** This is the first step where the raw data (documents) are loaded into the system. The documents could be in various formats such as text files, PDFs, HTML, etc. 
  * **Document Splitting:** Once the documents are loaded, they are split into smaller chunks or segments. This is done to make the data more manageable and to improve the efficiency of the subsequent steps.
  * **Vectors and Embeddings:** Each chunk of data is then transformed into a vector representation, also known as an embedding. These embeddings capture the semantic meaning of the data and are used for efficient retrieval of relevant information. LangChain primarily supports indexes and retrieval mechanisms centered around vector databases.
  * **Retrieval:** This is the final step where a user’s query is taken and the system uses the index to identify and return the most relevant documents. The retrieval is based on the similarity between the query vector and the document vectors.

![Indexes](img/index.png)

---

#### **Document Loading**

**Loading Environment Variable**

```python
from secret_key import hugging_facehub_key
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hugging_facehub_key
```

---

##### **PDF**

Load PDF using `pypdf` into array of documents, where each document contains the page content and metadata with `page` number

**Module Imports**

```python
from langchain.document_loaders import PyPDFLoader
from langchain import HuggingFaceHub
```

* `PyPDFLoader:` A module from LangChain used for loading PDF documents.

* `HuggingFaceHub:` A module from LangChain used for accessing pre-trained language models from the Hugging Face model hub.

**PDF Loading**

```python
# Load PDF
loader = PyPDFLoader("MachineLearning-Lecture01.pdf")
pages = loader.load()
```

* `loader: `An instance of ``PyPDFLoader`` initialized with the PDF file `"MachineLearning-Lecture01.pdf"`.

* `pages:` A list containing the extracted pages from the loaded PDF document.

---

##### **youtube**

**Install required packages**

```python
# ! pip install yt_dlp
# ! pip install pydub
# ! pip install openai
```

These lines are comments indicating that you need to install specific packages (`yt_dlp`, `pydub`, and `openai`). You can install them using pip if you haven't already done so.

```python
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

# Define the YouTube video URL
url = "https://www.youtube.com/watch?v=jGwO_UgTS7I"
# Define the directory to save the downloaded content
save_dir = "../docs/youtube/"
# Initialize the loader with YouTubeAudioLoader and OpenAIWhisperParser
loader = GenericLoader(
    YoutubeAudioLoader([url], save_dir),
    OpenAIWhisperParser()
)
docs = loader.load()
```

* Here, we import necessary modules from LangChain for loading documents from various sources. Specifically, we import `GenericLoader` for loading documents, `OpenAIWhisperParser` for parsing text, and `YoutubeAudioLoader` for loading audio from YouTube.

* These lines define the YouTube video URL and the directory to save the downloaded content from the video.

* Here, we initialize the document loader with `YoutubeAudioLoader` for loading audio content from the specified URL and `OpenAIWhisperParser` for parsing the audio content.

* We load the documents using the initialized loader, which downloads the audio content from the YouTube video, transcribes it, and parses it into documents.


**ffmpeg error clear method**

1. **Open Command Prompt as Administrator**

* Press `Win + X` and select `Command Prompt (Admin)` or `Windows PowerShell (Admin)`.

2. **Install Chocolatey:**

* Copy and paste the installation command for Chocolatey:

```sh
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

* Press `Enter` and wait for the installation to complete.

3. **Install FFmpeg using Chocolatey:**

* In the same command prompt (or a new one with administrative privileges), type:

```sh
choco install ffmpeg
```

* Press `Enter` and wait for Chocolatey to download and install FFmpeg.

4. **Verify FFmpeg Installation:**

* Open a new command prompt (not necessarily with administrative privileges) and type:

```sh
ffmpeg -version
```

* You should see the `FFmpeg version` information, confirming the installation is successful.

---

##### **URL`s**

```python
from langchain.document_loaders import WebBaseLoader

# Initialize the WebBaseLoader with the URL
loader = WebBaseLoader("https://github.com/tamaraiselva/git-demo/blob/main/metriales.docx")

# Load documents
docs = loader.load()

```

* This line imports the `WebBaseLoader` from LangChain, which is used to load documents from a web URL.

* Here, we initialize a `WebBaseLoader` object with the URL of a document hosted on the web. This loader will fetch the document from the specified URL.

* This line loads the document(s) using the initialized loader.

---

##### **NOTION**

* This line imports the `NotionDirectoryLoader` class from the `document_loaders` module in the LangChain framework. This loader is specifically designed to load documents from a directory containing Notion-exported Markdown files.

```python
from langchain.document_loaders import NotionDirectoryLoader

loader = NotionDirectoryLoader("notion")
docs = loader.load()

if docs:
    print(docs[0].page_content[0:100])
    print(docs[0].metadata)
else:
    print("No Notion documents were loaded.")
```

* Here, we create an instance of the `NotionDirectoryLoader` class and provide the path to the directory where Notion-exported Markdown files are located. In this case, the directory is named "notion".

* We use the `load()` method of the `loader` instance to load the documents from the specified directory. This method returns a list of `Document` objects representing the loaded documents.

* This conditional statement checks if any documents were loaded. If there are documents, it prints the first 100 characters of the content of the first document (`docs[0].page_content[0:100]`) and the metadata of the first document (`docs[0].metadata`). If no documents were loaded, it prints a message indicating that no Notion documents were loaded.

---

#### **Documnet Splitting**

**Loading Environment Variable**

```python
from secret_key import hugging_facehub_key
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hugging_facehub_key
```

**Text Splitting**

Here, we import the `CharacterTextSplitter` module from LangChain. This module provides functionality to split text into smaller chunks based on characters.

```python
from langchain.text_splitter import CharacterTextSplitter

chunk_size =26
chunk_overlap = 4

# Initialize the CharacterTextSplitter
c_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separator=' '  # Optional, if you want to split by a separator
)

# Define the text
text = 'abcdefghijklmnopqrstuvwxyzabcdefg'

# Split the text using the CharacterTextSplitter
chunks = c_splitter.split_text(text)

print(chunks)
```

* We define the parameters for text splitting.` chunk_size` specifies the maximum length of each chunk, and `chunk_overlap` specifies how much overlap there should be between adjacent chunks.

* We create an instance of `CharacterTextSplitter` and initialize it with the specified parameters. Optionally, you can specify a `separator` if you want to split the text based on a particular character or string.

* We define the text that we want to split into smaller chunks.

* We use the `split_text()` method of the `CharacterTextSplitter` instance to split the text into smaller chunks based on the specified parameters

* Finally, we print the resulting chunks.

**Recursive splitting details**

**`Using RecursiveCharacterTextSplitter:`**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, NotionDirectoryLoader

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", " ", ""]
)
```

- `Purpose:` This initializes a text splitter that recursively splits text into chunks based on defined parameters.

- `Parameters:`

  * `chunk_size:` The maximum size of each chunk in characters.

  * `chunk_overlap:` The overlap between consecutive chunks (in characters).

  * `separators:` List of strings that define separators used for splitting the text. Empty strings represent no separation.

- `Usage:` Useful for splitting long pieces of text into manageable chunks for processing.


**`Using CharacterTextSplitter:`**

```python
c_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150
)
```

- `Purpose:` This initializes a text splitter that splits text into chunks based on a single separator.

- `Parameters:`

  * `separator:` The string used to split the text into chunks.

  * `chunk_size:` The maximum size of each chunk in characters.

  * `chunk_overlap:` The overlap between consecutive chunks (in characters).

- `Usage:` Suitable for simpler text splitting tasks where a single separator suffices.

**`Using PyPDFLoader`**

```python
loader = PyPDFLoader("path_to_pdf_file")
pages = loader.load()
```

- `Purpose:` This loads a PDF document and extracts its pages.

- `Parameters:`

  * `"path_to_pdf_file":` The path to the PDF file.

- `Returns:` A list of page objects representing the contents of each page in the PDF.

**`Using NotionDirectoryLoader:`**

```python
loader = NotionDirectoryLoader("directory_path")
pages = loader.load()
```

- `Purpose:` This loads documents from a directory, assuming they are stored in Notion's format.

- `Parameters:`

  * `"directory_path":` The path to the directory containing Notion documents.

- `Returns:` A list of document objects representing the contents of each document in the directory.

**`Splitting Documents`**

```python
docs = c_splitter.split_documents(pages)
```

- `Purpose:` This splits loaded documents into chunks using the defined text splitter.

- `Parameters:`

  * `pages:` The list of document objects to be split.

- `Returns:` A list of document chunks, each representing a portion of the original document.

---

#### **Vectors and Embeddings**

**Embeddings**

LangChain Embeddings are numerical representations of text data, designed to be fed into machine learning algorithms. These embeddings are crucial for a variety of natural language processing (NLP) tasks, such as sentiment analysis, text classification, and language translation.

![image](img/e.jpg)

```python
# !pip install sentence-transformers
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()

text = "This is a test document to check the embeddings."
text_embedding = embeddings.embed_query(text)

print(f'Embeddings lenght: {len(text_embedding)}')
print (f"Here's a sample: {text_embedding[:5]}...")
```

* This line is a comment indicating that you should install the sentence-transformers package if you haven't already. It's likely included as a reminder in case the package isn't installed in your environment.

* Here, we import two different types of embedding models from LangChain: `OpenAIEmbeddings` and `HuggingFaceEmbeddings`. These models are used to generate embeddings for text.

* We initialize an embedding model using`HuggingFaceEmbeddings()`. This creates an instance of the Hugging Face embedding model.

* We define a sample text that we want to generate embeddings for.

* We use the initialized embedding model (`embeddings`) to generate embeddings for the given text (`text`) using the `embed_query()` method.

* We print the length of the embeddings generated for the text and show a sample of the embeddings. The length indicates the dimensionality of the embeddings, and the sample provides a glimpse of the first few values of the embeddings.

**Vectorstore**

VectorStore is a component of LangChain that facilitates efficient storage and retrieval of document embeddings, which are vector representations of documents. These embeddings are created using language models and are valuable for various natural language processing tasks such as information retrieval and document similarity analysis.


![image](img/v.jpg)

**Installation:**

To install VectorStore, you can use pip:

```python
# ! pip install langchain-chroma
from langchain_chroma import Chroma

db = Chroma.from_documents(splits, embeddings)

print(db._collection.count())
```

* First, import the Chroma class from langchain_chroma module.

* Then, create a VectorStore instance using the from_documents method. This method requires two parameters:

  * ` splits: `A list of document splits, where each split represents a document.

  * `embeddings:` A list of embeddings corresponding to the document splits.

* Finally, you can access the number of documents stored in the VectorStore using the count() method on the _collection attribute.

---

#### **Retrieval**

**Vectorstore retrieval**

```python
# !pip install lark
# !pip install pypdf tiktoken faiss-cpu

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

loader = PyPDFLoader("MachineLearning-Lecture01.pdf")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Initialize the HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(texts, embeddings)

# You can also specify search kwargs like k to use when doing retrieval.
#retriever = db.as_retriever()
retriever = db.as_retriever(search_kwargs={"k": 2})

print(len(documents))
```

* This code block is commented out, but it suggests installing necessary packages using pip. However, since it's commented out, it doesn't affect the execution of the code. These packages seem to be dependencies for LangChain.

* These lines import necessary modules from LangChain for document loading (`PyPDFLoader`), text splitting (`CharacterTextSplitter`, `RecursiveCharacterTextSplitter`), vector stores (`FAISS`), and embeddings (`HuggingFaceEmbeddings`).

* Here, a `PyPDFLoader` instance is created to load a PDF document named "MachineLearning-Lecture01.pdf". The `load()` method is then used to extract the content of the document into a list of documents.

* A `CharacterTextSplitter` instance is created with specified parameters for chunk size and overlap. Then, the `split_documents()` method is used to split the documents into smaller text chunks based on the specified parameters.

* An instance of `HuggingFaceEmbeddings` is initialized. Then, a FAISS vector store (`FAISS`) is created from the text chunks using the embeddings obtained from the Hugging Face model.

* A retriever object is created from the FAISS vector store. Additional search arguments, such as the number of nearest neighbors (`k`), can be specified.

* This line prints the number of documents loaded from the PDF file.

---

### **Memory**

Memory which is still in beta phase is an essential component in a conversation. This allows us to infer information in past conversations. Users have various options, including preserving the complete history of all conversations, summarizing the ongoing conversation, or retaining the most recent n exchanges.

![image](img/memory.png)

**Loading Environment Variable**

```python
from secret_key import hugging_facehub_key
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hugging_facehub_key
```

**Models**

```python
import datetime
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2024,5,5):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
print(llm_name)
```

#### **Chat Message History**

```python
from langchain.memory import ChatMessageHistory
from langchain import PromptTemplate, LLMChain

# Initialize chat message history
history = ChatMessageHistory()

# Add user and AI messages to the history
history.add_user_message("hi!")
history.add_ai_message("whats up?")

# Print the current messages in the history
print(history.messages)

# Add another user message and print the updated history
history.add_user_message("Fine, what about you?")
print(history.messages)

# Define a prompt template with a question variable
template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Create an LLMChain with the prompt template and a language model
# Note: 'llm' should be initialized before using it here
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Define a question and run the chain to get the model's response
question = "Which city does the company's headquarters for our international employees reside in?"
print(llm_chain.run(question))
```

* This code block imports necessary modules from LangChain for managing chat message history (`ChatMessageHistory`), creating prompt templates (`PromptTemplate`), and executing language model chains (`LLMChain`).

* An instance of `ChatMessageHistory` is initialized to manage the chat message history.

* We add a message from the user to the chat history with the content "hi!". 

* We add a message from the AI to the chat history with the content "whats up?".

* We print the current messages in the chat history.

* We add another message from the user to the chat history with the content "Fine, what about you?" and then display the updated chat messages.

* We define a template for prompting the language model. The template includes a placeholder {*question*} for the user's question and a fixed part "Answer: Let's think step by step." to guide the model's response.

* We create a `PromptTemplate` instance using the defined template. The `input_variables` parameter specifies that the template expects a variable named `question`.

* We create an `LLMChain` instance using the `prompt` and a language model instance `llm`. This chain will use the prompt template to generate responses based on the input question.

* We define a question to be asked to the language model: "Which city does the company's headquarters for our international employees reside in?" and use the `llm_chain` to generate and print the model's response.

---

#### **Conversation Buffer Memory**

* This method involves stacking all user-agent interactions into the prompt, allowing the model to track the entire conversation. However, it poses limitations in handling long conversations due to token span constraints.

```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)

conversation.predict(input="Hi, my name is ram")
conversation.predict(input="What is 1+1?")
conversation.predict(input="What is my name?")
print(memory.buffer)
memory.load_memory_variables({})
memory = ConversationBufferMemory()
memory.save_context({"input": "Hi"}, 
                    {"output": "What's up"})
print(memory.buffer)
memory.load_memory_variables({})
memory.save_context({"input": "Not much, just hanging"}, 
                    {"output": "Cool"})
memory.load_memory_variables({})
```

1. **Importing Necessary Modules**

    * Here, we import modules required for setting up a conversation chain and managing conversation memory.

2. **Setting up Conversation Memory**

    * This line initializes a conversation memory buffer, which will store the context and history of the conversation.

3. **Creating a Conversation Chain**

    * A conversation chain is created, which uses a language model (llm) and the initialized conversation memory buffer. Setting verbose=True enables verbose mode, providing additional information during conversation processing

4. **Interacting with the Conversation Chain**

    * These lines simulate a conversation by providing inputs to the conversation chain. Each input triggers a response from the language model based on the conversation context.

5. **Accessing Conversation Memory**

    * This line prints the current content of the conversation memory buffer, which includes the conversation history and context.

6. **Managing Conversation Memory**

    * These lines demonstrate managing conversation memory. `load_memory_variables({})` clears the memory buffer, while `ConversationBufferMemory()` creates a new instance of the conversation memory.

    * These lines save and load conversation context into the memory buffer **save_context()** stores input-output pairs in the memory, while **load_memory_variables({})** resets the memory variables.

---

#### **Conversation Buffer Window Memory**

* This approach limits the memory to the last few interactions, addressing token constraints while still retaining recent context. It's a compromise between full conversation tracking and token efficiency.

```python
from langchain.memory import ConversationBufferWindowMemory

# Initialize ConversationBufferWindowMemory with k=1
memory = ConversationBufferWindowMemory(k=1)

# Save conversation contexts
memory.save_context({"input": "Hi"},
                    {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})

# Load memory variables (optional)
memory.load_memory_variables({})

# Reinitialize memory (optional, seems redundant)
memory = ConversationBufferWindowMemory(k=1)

# Create ConversationChain object
conversation = ConversationChain(
    llm=llm, 
    memory=memory,
    verbose=False
)

# Predict responses to inputs
conversation.predict(input="Hi, my name is ram")
conversation.predict(input="What is 1+1?")
conversation.predict(input="What is my name?")
```

* We import the ConversationBufferWindowMemory class from the LangChain library. This class represents a memory module that stores conversation contexts.

* We initialize a ConversationBufferWindowMemory instance with a window size of 1 (k=1). This means that the memory will retain the most recent conversation context.

* We save two conversation contexts into the memory. Each context consists of an input and an output.

* We load memory variables, although it appears to be redundant in this context as no additional parameters are provided.

* This line reinitializes the memory object with the same parameters as before. However, it seems redundant since the memory was already initialized earlier.

* We create a `ConversationChain` object, passing in a language model (`llm`), the initialized memory object, and setting verbose to False.

* We predict responses to three different inputs using the `ConversationChain` object. Each input represents a message in a conversation.

---

#### **Conversation Token Buffer Memory**

```python
#!pip install tiktoken
from langchain.memory import ConversationTokenBufferMemory
# Initialize memory
memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50)

# Save contexts
memory.save_context({"input": "AI is what?!"}, {"output": "Amazing!"})
memory.save_context({"input": "Backpropagation is what?"}, {"output": "Beautiful!"})
memory.save_context({"input": "Chatbots are what?"}, {"output": "Charming!"})

# Load memory variables
memory.load_memory_variables({})

```

* `Installation:` First, ensure you have the tiktoken library installed by running !pip install tiktoken.

* `Importing Modules:` Import the necessary module from LangChain:

    * This imports the ConversationTokenBufferMemory class, which is used to manage conversation tokens in a buffer memory.

* `Initializing Memory:` Create an instance of `ConversationTokenBufferMemory`:

    * Initialize the memory object with the language model (llm) and a maximum token limit of 50.

* `Saving Contexts:` Save conversation contexts into the memory:

  * Each `save_context` call saves an input-output pair into the memory. For example, the input "AI is what?!" corresponds to the output "Amazing!".

* `Loading Memory Variables:` Load memory variables to reset the memory:

  * Each save_context call saves an input-output pair into the memory. For example, the input "AI is what?!" corresponds to the output "Amazing!". 

* To document this code for clarity:

  * `Purpose:` The code manages conversation tokens in a buffer memory, allowing the storage and retrieval of input-output pairs.

  * `Initialization:` The ConversationTokenBufferMemory class is initialized with a language model and a maximum token limit.

  * **Functionality:**

    * `Saving Contexts:` Saves input-output pairs into the memory.

    * `oad_memory_variables:` Loads memory variables, potentially for configuration or additional data.

---

#### **Conversation Summary Memory**

* This method summarizes the conversation at each step, reducing token usage but maintaining context. It provides a balance between retaining conversation history and token efficiency.

```python
from langchain.memory import ConversationSummaryBufferMemory
# create a long string
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"}, 
                    {"output": f"{schedule}"})
memory.load_memory_variables({})
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)
conversation.predict(input="What would be a good demo to show?")
memory.load_memory_variables({})
```

* We import the `ConversationSummaryBufferMemory` class from the LangChain memory module. This class allows us to store and manage conversational context and summaries.

* We create a long string named `schedule` that contains a detailed schedule for the day, including meetings, work on a project, lunch plans, and more.

* We initialize a `ConversationSummaryBufferMemory` object named `memory`. This object will store conversational context and summaries. We pass the language model (`llm`) and a maximum token limit (`max_token_limit`) as parameters.

* We save conversational contexts and their corresponding summaries into the memory object. Each context consists of an input (what was said in the conversation) and an output (the summary or response). Here, we save three different conversational contexts and their corresponding summaries.

* We load the memory variables into the memory object. This step ensures that the memory is ready for use in the conversation chain.

* We create a `ConversationChain` object named `conversation` that uses the language model (`llm`) and the initialized memory object. The `verbose=True` parameter enables verbose mode for additional information during conversation processing.

* We predict a response to the input "What would be a good demo to show?" using the `ConversationChain` object. This input triggers the model to generate a response based on the conversation context and the stored summaries.

---

#### **Knowledge Graph Memory**

* Utilizes a KG structure to extract relevant information from the conversation, organizing it into entities and relationships. It enhances contextual understanding and facilitates downstream tasks.

```python
from langchain.memory import ConversationKGMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain import HuggingFaceHub
import networkx as nx
import matplotlib.pyplot as plt

memory = ConversationBufferWindowMemory(k=1)  

# Define LLM
repo_id = "google/flan-t5-large"
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64}
)

template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know. The AI ONLY uses information contained in the "Relevant Information" section and does not hallucinate.

Relevant Information:

{history}

Conversation:
Human: {input}
AI:"""
prompt = PromptTemplate(input_variables=["history", "input"], template=template)

conversation_with_kg = ConversationChain(
    llm=llm, 
    verbose=True, 
    prompt=prompt,
    memory=ConversationKGMemory(llm=llm)
)


conversation_with_kg.predict(input="Hi there! I am Sam")
conversation_with_kg.predict(input="My TV is broken and I need some customer assistance")
conversation_with_kg.predict(input="Yes it is and it is still under warranty. my warranty number is A512423")

# nx.draw(conversation_with_kg.memory.kg, with_labels=True)
# plt.show()

print(conversation_with_kg.memory.kg)
print(conversation_with_kg.memory.kg.get_triples())
```

**1. Importing Required Modules**

* We import necessary modules from LangChain for memory management and Hugging Face integration, as well as NetworkX and Matplotlib for handling and visualizing the knowledge graph.

**2. Setting Up the Language Model**

* We define a language model (LLM) using the `google/flan-t5-large` model from Hugging Face. Model-specific parameters like `temperature` and `max_length` are also specified.

**3. Creating the Prompt Template**


* We create a prompt template that guides the conversation between the human and the AI. The AI is instructed to use only the information provided in the "Relevant Information" section and avoid fabricating answers.

**4. Setting Up the Conversation Chain**

* We set up a conversation chain using the defined LLM, the prompt template, and a `ConversationKGMemory` for maintaining a knowledge graph memory.

**5. Running the Conversation**

* We simulate a conversation where the user interacts with the AI, providing different inputs. The AI uses the knowledge graph memory to maintain context and provide relevant responses.

**6. Visualizing the Knowledge Graph**

* This part of the code (commented out) is for visualizing the knowledge graph using NetworkX and Matplotlib. If uncommented, it would draw the graph and display it.

**7. Printing the Knowledge Graph and Its Triples**

* Finally, we print the knowledge graph object and its triples (relationships between entities) to see the structure of the knowledge graph.

---

#### **Entity Memory**

* Extracts specific entities mentioned in the conversation, such as names, numbers, or keywords, for further processing or response generation. It aids in understanding user intent and context.

```python
from langchain.memory import ConversationEntityMemory
from langchain.chains.conversation import ConversationChain
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain import HuggingFaceHub
from pprint import pprint

# Define LLM
repo_id = "google/flan-t5-large"  # The ID of the pre-trained model on Hugging Face
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64}
)

# Initialize ConversationEntityMemory
memory = ConversationEntityMemory(llm=llm)

# Initialize ConversationChain
conversation = ConversationChain(
    llm=llm, 
    verbose=True,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=memory
)

# Make predictions and interact with memory
conversation.predict(input="Hi I am Sam. My TV is broken but it is under warranty.")
conversation.predict(input="How can I get it fixed. The warranty number is A512453")

# Access the entity cache
conversation.memory.entity_cache

conversation.predict(input="Can you send the repair person call Dave to fix it?")
conversation.memory.entity_cache


# Pretty print the memory object
pprint(conversation.memory)
```

* We import the necessary modules and functions. `ConversationEntityMemory` and `ConversationChain` are used to manage and conduct conversations with entity memory. `ENTITY_MEMORY_CONVERSATION_TEMPLATE` is a predefined prompt template. `HuggingFaceHub` allows us to use a pre-trained model from Hugging Face, and `pprint` is used for pretty-printing objects for better readability.This code snippet demonstrates the use of the `ConversationEntityMemory` class in LangChain to manage entities and their relationships in a conversation.

* We define the language model (LLM) using the `HuggingFaceHub` class. We specify the model ID (`google/flan-t5-large`) and provide model-specific arguments (`temperature` and `max_length`). The `temperature` controls the randomness of the output, and `max_length` specifies the maximum length of the generated text.

* We initialize a `ConversationEntityMemory` object named `memory`. This memory object will store entities and their relationships during the conversation.

* We create a `ConversationChain` object named `conversation` that uses the language model (`llm`), enables verbose mode, specifies a predefined prompt template (`ENTITY_MEMORY_CONVERSATION_TEMPLATE`), and uses the initialized memory object.

* We simulate a conversation by providing inputs to the conversation chain. The AI responds based on the conversation context and the entity memory.

* We access the entity cache stored in the memory object to view the entities and their relationships at different points in the conversation.

* We pretty-print the memory object to display the entities and relationships stored in the entity cache.

---

### **Chains**

* Chains form the backbone of LangChain's workflows, seamlessly integrating Language Model Models (LLMs) with other components to build applications through the execution of a series of functions.

* The fundamental chain is the LLMChain, which straightforwardly invokes a model and a prompt template. For example, consider saving a prompt as "ExamplePrompt" and intending to run it with Flan-T5. By importing LLMChain from langchain.chains,

* `you can define a chain_example like so:` LLMChain(llm=flan-t5, prompt=ExamplePrompt). Executing the chain for a given input is as simple as calling chain_example.run("input").

* For scenarios where the output of one function needs to serve as the input for the next, SimpleSequentialChain comes into play. Each function within this chain can employ diverse prompts, tools, parameters, or even different models, catering to specific requirements.

![image](img/chain.png)

> Chains are simple objects that essentially string together several components (for linear pipelines).

**EXAMPLE**

```python
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain

# Article text
article = '''Coinbase, the second-largest crypto exchange by trading volume, released its Q4 2022 earnings on Tuesday, giving shareholders and market players alike an updated look into its financials. In response to the report, the company's shares are down modestly in early after-hours trading.In the fourth quarter of 2022, Coinbase generated 
2.49 billion in the year-ago quarter. Coin base's top line was not enough to cover its expenses: The company lost 
2.46 per share, and an adjusted EBITDA deficit of 
581.2 million in revenue and earnings per share of -
201.8 million driven by 8.4 million monthly transaction users (MTUs), according to data provided by Yahoo Finance.Before its Q4 earnings were released, Coinbase's stock had risen 86% year-to-date. Even with that rally, the value of Coinbase when measured on a per-share basis is still down significantly from its 52-week high of 
26 billion in the third quarter of last year to 
133 billion to 
1.5 trillion during 2022, which resulted in Coinbase's total trading volumes and transaction revenues to fall 50% and 66% year-over-year, respectively, the company reported.As you would expect with declines in trading volume, trading revenue at Coinbase fell in Q4 compared to the third quarter of last year, dipping from 
322.1 million. (TechCrunch is comparing Coinbase's Q4 2022 results to Q3 2022 instead of Q4 2021, as the latter comparison would be less useful given how much the crypto market has changed in the last year; we're all aware that overall crypto activity has fallen from the final months of 2021.)There were bits of good news in the Coinbase report. While Coinbase's trading revenues were less than exuberant, the company's other revenues posted gains. What Coinbase calls its "subscription and services revenue" rose from 
282.8 million in Q4 of the same year, a gain of just over 34% in a single quarter.And even as the crypto industry faced a number of catastrophic events, including the Terra/LUNA and FTX collapses to name a few, there was still growth in other areas. The monthly active developers in crypto have more than doubled since 2020 to over 20,000, while major brands like Starbucks, Nike and Adidas have dived into the space alongside social media platforms like Instagram and Reddit.With big players getting into crypto, industry players are hoping this move results in greater adoption both for product use cases and trading volumes. Although there was a lot of movement from traditional retail markets and Web 2.0 businesses, trading volume for both consumer and institutional users fell quarter-over-quarter for Coinbase.Looking forward, it'll be interesting to see if these pieces pick back up and trading interest reemerges in 2023, or if platforms like Coinbase will have to keep looking elsewhere for revenue (like its subscription service) if users continue to shy away from the market.
'''

# Create prompt for extracting facts
fact_extraction_prompt = PromptTemplate(
    input_variables=["text_input"],
    template="Extract the key facts out of this text. Don't include opinions. Give each fact a number and keep them short sentences. :\n\n {text_input}"
)

# Initialize fact extraction chain
fact_extraction_chain = LLMChain(llm=llm, prompt=fact_extraction_prompt)

# Extract facts
facts = fact_extraction_chain.run(article)
print(facts)

# Create prompt for investor update
investor_update_prompt = PromptTemplate(
    input_variables=["facts"],
    template="Write an investor update using these key facts:\n\n {facts}"
)

# Initialize investor update chain
investor_update_chain = LLMChain(llm=llm, prompt=investor_update_prompt)

# Generate investor update
investor_update = investor_update_chain.run(facts)
print(investor_update)

# Create prompt for knowledge graph triples
triples_prompt = PromptTemplate(
    input_variables=["facts"],
    template="Take the following list of facts and turn them into triples for a knowledge graph:\n\n {facts}"
)

# Initialize triples chain
triples_chain = LLMChain(llm=llm, prompt=triples_prompt)

# Generate triples
triples = triples_chain.run(facts)
print(triples)
```

* We import the necessary components from LangChain: HuggingFaceHub for accessing the language model, PromptTemplate for defining prompts, and LLMChain for creating chains of operations with the language model.

**1. Article Text**

* Here, we define a long article text about Coinbase's financial performance and developments in the crypto market.

**2. Fact Extraction Prompt**

* We create a prompt template for extracting key facts from the article text. The template instructs the model to extract facts without including opinions and present each fact as a numbered short sentence.

* We set up an LLMChain with the fact extraction prompt, which combines the language model (llm) with the prompt.

* We run the chain on the article to extract key facts, storing the result in the facts variable.

**3. Investor Update Prompt**

* Next, we define a prompt for generating an investor update using the extracted facts. We create another LLMChain with this prompt.

* We run the chain on the extracted facts to generate an investor update based on the key facts.

**4. Knowledge Graph Triples Prompt**

* We create a prompt template for converting the extracted facts into triples for a knowledge graph. The template instructs the model to transform the facts into structured triples.

* We run the triples chain using the extracted facts to generate knowledge graph triples.

**5. Print Results**

* Finally, we print the extracted facts, the generated investor update, and the knowledge graph triples to see the outputs of each step in the chain.

---

### **Agents**

* Agents, at their core, leverage a language model to make decisions about a sequence of actions to be taken. Unlike chains where a predefined sequence of actions is hard coded directly in the code, agents use a llm as a reasoning engine to determine the actions to be taken and their order.

![agent](img/agent.svg)

  > Agents are more sophisticated, allowing business logic to let you choose how the components should interact

**EXAMPLE**

```python
# !pip install google-search-results

from secret_key import hugging_facehub_key
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hugging_facehub_key
os.environ['SERPAPI_API_KEY'] = "4a03df4b92e18dc16b4ac554d979aac6275516dd94270dbaf8a09e3f8ce956e4"

# Import necessary modules
from langchain_community.llms import HuggingFaceHub
from langchain.agents import initialize_agent, Tool, AgentType
from langchain import SerpAPIWrapper

# See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.5, "max_length": 64}
)
# Define the SerpAPIWrapper tool
search_tool = Tool(
    name="Intermediate Answer",
    func=SerpAPIWrapper().run,
    description="Useful for when you need to ask with search"
)

# Initialize the agent with the HuggingFaceHub model and SerpAPIWrapper tool
self_ask_with_search_agent = initialize_agent(
    [search_tool], 
    llm, 
    agent=AgentType.SELF_ASK_WITH_SEARCH,
    return_intermediate_steps=True,
    verbose=True
)

# Run the agent with a question
self_ask_with_search_agent("What is the hometown of the reigning men's French Open?")

self_ask_with_search_agent("What is the 10th fibonacci number?")
```

1. **Environment Setup:**

* Explain the installation of necessary packages (e.g., google-search-results).

* Describe the importation of the secret key and setting up environment variables.

2. **Model Initialization:**

* Explain the process of initializing the Hugging Face model, specifying the model chosen and the parameters used.
Tool Definition:

* Describe the creation of the SerpAPIWrapper tool, its purpose, and how it integrates with the agent.

3. **Agent Initialization:**

* Detail the steps to initialize the agent with the chosen LLM and search tool.

* Explain the significance of the agent type `(SELF_ASK_WITH_SEARCH)` and the parameters (`return_intermediate_steps` and `verbose`).

4. **Running the Agent:**

* Demonstrate how to use the agent to answer questions.

* Provide examples of questions asked and explain the expected outputs.
---

## **Embeddings**

* Simply put, text embeddings are a method of translating words or sentences from human language into the language of computers — numbers. This numerical representation of text allows machines to understand and process language meaningfully, enabling many of the advanced NLP applications we use today.

![embeddings](img/Embedding.webp)

**Step 1: Using Hugging Face Transformers to Embed Sentences**

* First, we utilize the transformers library from Hugging Face to embed sentences using a pre-trained BERT model.

**1. Load the tokenizer and model:**

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Choose a pre-trained model
model_name = 'bert-base-uncased'

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
```

**2. Define sentences and tokenize them:**

```python
# Define the sentences you want to embed
sentences = ["This is a sample sentence.", "This is another example."]

# Tokenize the sentences
inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
```

**3.Print tokenized inputs:**

```python
# Verify the structure of inputs
print("Tokenized Inputs:", inputs)
print("input_ids:", inputs['input_ids'])
print("attention_mask:", inputs['attention_mask'])
```

**4. Generate embeddings using the model:**

```python
# Generate embeddings
with torch.no_grad():
    outputs = model(**inputs)

# Typically, use the [CLS] token's embedding (first token) for sentence-level tasks
embeddings = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
print("Transformers Embeddings:\n", embeddings)
```

**Step 2: Using Sentence-Transformers to Embed Sentences**

* Next, we use the sentence-transformers library, which is designed to produce better sentence embeddings.

![embeddings](img/Embedding1.png)

**1. Load the SentenceTransformer model:**

```python
from sentence_transformers import SentenceTransformer

# Load a pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # This is a smaller, faster model suitable for embeddings
```

**2. Encode sentences and print embeddings:**

```python
# Encode sentences to get their embeddings
embeddings = model.encode(sentences)
print("Sentence-Transformers Embeddings:\n", embeddings)
```

**Step 3: Encoding and Comparing Specific Sentences**

* We proceed to encode specific sentences and compare their embeddings.

**1. Encode new sentences:**

```python
emb_1 = model.encode(["What is the meaning of life?"])
emb_2 = model.encode(["How does one spend their time well on Earth?"])
emb_3 = model.encode(["Would you like a salad?"])

print("Sentence-Transformers Embeddings:\n", emb_1)
print("Sentence-Transformers Embeddings:\n", emb_2)
print("Sentence-Transformers Embeddings:\n", emb_3)
```

**2. Print sample sentences and phrases:**

```python
in_1 = "The kids play in the park."
in_2 = "The play was for kids in the park."
print(in_1)
print(in_2)
in_pp_1 = ["kids", "play", "park"]
in_pp_2 = ["play", "kids", "park"]
```

**3. Encode phrases and calculate their mean embeddings:**

```python
embeddings_1 = model.encode(in_pp_1)
embeddings_2 = model.encode(in_pp_2)
import numpy as np
emb_array_1 = np.stack(embeddings_1)
print(emb_array_1.shape)
emb_array_2 = np.stack(embeddings_2)
print(emb_array_2.shape)
emb_2_mean = emb_array_2.mean(axis=0)
emb_1_mean = emb_array_1.mean(axis=0)

print(emb_1_mean.shape)
print(emb_2_mean.shape)

print(emb_1_mean[:4])
print(emb_2_mean[:4])
```

**4. Encode sentences and print their first 4 values:**

```python
embedding_1 = model.encode([in_1])
embedding_2 = model.encode([in_2])

vector_1 = embedding_1[0]
print("Vector 1 first 4 values:", vector_1[:4])
vector_2 = embedding_2[0]
print("Vector 2 first 4 values:", vector_2[:4])
```

---

## **Chain of Thought**

* Chain-of-thought prompting is a prompt engineering technique that aims to improve language models' performance on tasks requiring logic, calculation and decision-making by structuring the input prompt in a way that mimics human reasoning.

* To construct a chain-of-thought prompt, a user typically appends an instruction such as "Describe your reasoning in steps" or "Explain your answer step by step" to the end of their query to a large language model (LLM). In essence, this prompting technique asks the LLM to not only generate an end result, but also detail the series of intermediate steps that led to that answer.

* Guiding the model to articulate these intermediate steps has shown promising results. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models," a seminal paper by the Google Brain research team presented at the 2022 NeurIPS conference, found that chain-of-thought prompting outperformed standard prompting techniques on a range of arithmetic, common-sense and symbolic reasoning benchmarks.

> Chain of Thought (CoT) prompting is a technique that helps Large Language Models (LLMs) perform complex reasoning tasks by breaking down the problem into a series of intermediate steps. Think of it as providing the LLM with a roadmap to follow instead of just the destination.

![chain-of-thought](img/cot.jpeg)

### **Chain-of-Thought (CoT)**

* Chain-of-Thought (CoT) prompting enhances reasoning in large language models(LLMs) by breaking down complex problems into smaller, manageable steps. This method mirrors how humans tackle complicated math or logic questions by decomposing them into intermediate steps that lead to a final answer.

* CoT prompting involves structuring the input prompt to guide the LLM to explain its reasoning in a step-by-step manner. This approach has shown significant improvements in LLM performance on various reasoning tasks, including arithmetic, common-sense, and symbolic reasoning.

* Researchers Wei et al. discovered that with the right examples, LLMs can also perform this step-by-step reasoning. This process, known as in-context learning, doesn't require adjusting the model's weights. By presenting the LLM with examples of how to break down problems, it learns to replicate this structured approach.

![chain-of-thought](img/cot.webp)

**Prompt:**

```python
The odd numbers in this group add up to an even number: 4, 8, 9, 15, 12, 2, 1.
A: Adding all the odd numbers (9, 15, 1) gives 25. The answer is False.
The odd numbers in this group add up to an even number: 17,  10, 19, 4, 8, 12, 24.
A: Adding all the odd numbers (17, 19) gives 36. The answer is True.
The odd numbers in this group add up to an even number: 16,  11, 14, 4, 8, 13, 24.
A: Adding all the odd numbers (11, 13) gives 24. The answer is True.
The odd numbers in this group add up to an even number: 17,  9, 10, 12, 13, 4, 2.
A: Adding all the odd numbers (17, 9, 13) gives 39. The answer is False.
The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1. 
A:
```

**Output:**

```python
Adding all the odd numbers (15, 5, 13, 7, 1) gives 41. The answer is False.
```

### **Few-Shot Prompting**

* While large-language models demonstrate remarkable zero-shot capabilities, they still fall short on more complex tasks when using the zero-shot setting. Few-shot prompting can be used as a technique to enable in-context learning where we provide demonstrations in the prompt to steer the model to better performance. The demonstrations serve as conditioning for subsequent examples where we would like the model to generate a response.

**Prompt:**

```python
A "whatpu" is a small, furry animal native to Tanzania. An example of a sentence that uses the word whatpu is:
We were traveling in Africa and we saw these very cute whatpus. 
To do a "farduddle" means to jump up and down really fast. An example of a sentence that uses the word farduddle is:
```

**Output:**

```python
When we won the game, we all started to farduddle in celebration.
```

### **Self-consistency Prompting**

* This approach combines diverse reasoning paths with few-shot CoT to find the answer with the highest consistency. This method performs well with arithmetic and commonsense reasoning problems. Self-consistency prompt samples a diverse set of reasoning paths instead of the greedy one and then finalizes the most consistent answer by marginalizing out the sampled reasoning paths.

![self-consistency](img/scot.png)

**Prompt:**

```python
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done,
there will be 21 trees. How many trees did the grove workers plant today?
A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted.
So, they must have planted 21 - 15 = 6 trees. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Leah had 32 chocolates and Leah’s sister had 42. That means there were originally 32 + 42 = 74
chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops
did Jason give to Denny?
A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of
lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does
he have now?
A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so
in total he has 7 + 2 = 9 toys. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from
monday to thursday. How many computers are now in the server room?
A: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 =
20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers.
The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many
golf balls did he have at the end of wednesday?
A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On
Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: She bought 5 bagels for $3 each. This means she spent $15. She has $8 left.

Q: When I was 6 my sister was half my age. Now I’m 70 how old is my sister?
A:
```

**Output 1:**

```python
When I was 6 my sister was half my age, so she was 3. Now I am 70, so she is 70 - 3 = 67. The answer is 67.
```

**Output 2:**

```python
When the narrator was 6, his sister was half his age, which is 3. Now that the narrator is 70, his sister would be 70 - 3 = 67 years old. The answer is 67.
```

**Output 3:**

```python
When I was 6 my sister was half my age, so she was 3. Now I am 70, so she is 70/2 = 35. The answer is 35.
```

### **Zero-Shot Prompting**

* Zero-shot prompting is a technique that allows large language models (LLMs) to generate responses to queries without any specific training on the task. By providing a prompt that describes the task or question, the LLM can generate an answer based on its pre-existing knowledge and understanding of language.

* Large language models (LLMs) today, such as GPT-3.5 Turbo, GPT-4, and Claude 3, are tuned to follow instructions and are trained on large amounts of data. Large-scale training makes these models capable of performing some tasks in a "zero-shot" manner. Zero-shot prompting means that the prompt used to interact with the model won't contain examples or demonstrations. The zero-shot prompt directly instructs the model to perform a task without any additional examples to steer it.

**Prompt:**

```python
Classify the text into neutral, negative or positive. 
Text: I think the vacation is okay.
Sentiment:
```

**Output:**

```python
Neutral
```

---

## **Retriever Augmented Generator**

* **Retrieval-Augmented Generation (RAG)** is an AI framework that combines the generative capabilities of large language models (LLMs) with external knowledge sources. It aims to enhance the quality of responses generated by LLMs by cross-referencing an authoritative knowledge base outside their training data. This approach ensures more accurate, relevant, and context-aware answers.

* The basic usage of an LLM consists of giving it a prompt and getting back a response.

![rag](img/llm.webp)

* RAG works by adding a step to this basic process. Namely, a retrieval step is performed where, based on the user’s prompt, the relevant information is extracted from an external knowledge base and injected into the prompt before being passed to the LLM.

![rag](img/rag.webp)

There are 2 key elements of a RAG system: a retriever and a knowledge base.

**Retriever**

* A retriever takes a user prompt and returns relevant items from a knowledge base. This typically works using so-called text embeddings, numerical representations of text in concept space. In other words, these are numbers that represent the meaning of a given text.

* Text embeddings can be used to compute a similarity score between the user’s query and each item in the knowledge base. The result of this process is a ranking of each item’s relevance to the input query.

* The retriever can then take the top k (say k=3) most relevant items and inject them into the user prompt. This augmented prompt is then passed into the LLM for generation.

![rag](img/rag.webp.png)

**Knowledge Base**

* The next key element of a RAG system is a knowledge base. This houses all the information you want to make available to the LLM. While there are countless ways to construct a knowledge base for RAG, here I’ll focus on building one from a set of documents.

* The process can be broken down into 4 key steps [2,3].

  1. **Load docs** This consists of gathering a collection of documents and ensuring they are in a ready-to-parse format (more on this later).

  2. **Chunk docs** Since LLMs have limited context windows, documents must be split into smaller chunks (e.g., 256 or 512 characters long).

  3. **Embed chunks** Translate each chunk into numbers using a text embedding model.

  4. **Load into Vector DB** Load text embeddings into a database (aka a vector database).

![rag](img/KOT.webp)

**EXAMPLE**

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# Import any embedding model from the Hugging Face hub
# You can choose an embedding model as per your requirement
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# Alternative embedding model
# Settings.embed_model = HuggingFaceEmbedding(model_name="thenlper/gte-large")

# Set LLM to None, assuming no language model is used directly here
Settings.llm = None

# Set chunk size and overlap for document chunking
Settings.chunk_size = 256
Settings.chunk_overlap = 25

# Load articles from the directory
# Replace "docs" with the actual path to your documents directory
documents = SimpleDirectoryReader("docs").load_data()

# Print the initial count of documents
print(f"Initial number of documents: {len(documents)}")

# Filter out documents based on specific criteria
documents = [doc for doc in documents if "Member-only story" not in doc.text and
             "The Data Entrepreneurs" not in doc.text and " min read" not in doc.text]

# Print the count of documents after filtering
print(f"Number of documents after filtering: {len(documents)}")

# Store documents into a vector database
index = VectorStoreIndex.from_documents(documents)

# Set the number of documents to retrieve
top_k = 3

# Configure the retriever with the index and similarity threshold
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=top_k,
)

# Assemble the query engine with the retriever and postprocessor
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
)

# Define the query
query = "What is fat-tailedness?"

# Get the response from the query engine
response = query_engine.query(query)

# Reformat and print the response
context = "Context:\n"
for i in range(min(top_k, len(response.source_nodes))):
    context += response.source_nodes[i].text + "\n\n"

print(context)
```
![rag](img/retriverrag.webp)

**1. Importing Libraries:**

* These lines import the necessary classes from the `llama_index` package. This includes embedding models, settings, data readers, vector stores, retrievers, query engines, and postprocessors.

**2. Setting Up the Embedding Model:**

* The code sets up the embedding model to be used for text embeddings. In this example, the `BAAI/bge-small-en-v1.5` model is chosen, but you can select a different model based on your requirements.

**3. Setting the Language Model:**

* Sets the language model (LLM) to None, indicating that no specific LLM is being used directly in this setup.

**4. Configuring Document Chunking:**

* `chunk_size` defines the size of text chunks into which the documents will be split.

* `chunk_overlap` defines the overlap between consecutive chunks to ensure context continuity.

**5. Loading Documents:**

* The code loads documents from a specified directory. You need to replace `"docs"` with the actual path to your documents directory.

**6. Print Initial Document Count:**

* Prints the number of documents loaded to verify the initial count.

**7. Filtering Documents:**

* Filters out specific documents based on criteria such as content or metadata. In this example, documents containing certain phrases are excluded.

**8. Print Filtered Document Count:**

* Prints the number of documents after filtering to confirm the updated count.

**9. Creating Vector Store Index:**

* Converts the documents into a vector store index, which facilitates efficient similarity search.

**10. Setting Top-k:**

* Defines how many top similar documents to retrieve in response to a query.

**11. Configuring Retriever:**

* Sets up the retriever with the vector index and specifies that it should return the top `k` most similar documents.

**12. Assembling Query Engine:**

* Configures the query engine using the retriever and a similarity postprocessor. The postprocessor filters out results below a similarity threshold of 0.5.

**13. Defining the Query:**

* Specifies the query string to be used for retrieving relevant documents.

**14. Querying the Documents:**

* Executes the query using the query engine and retrieves the response containing the most relevant documents.

**15. Formatting and Printing the Response:**

* Formats the response by concatenating the text of the retrieved documents up to the specified `top_k`limit.

* Prints the formatted context, providing a readable output of the retrieved document snippets.

---

## **Advanced Retrieval Techniques:**

* RAG (Retrieval Augmented Generation) is a technique for augmenting LLM knowledge with additional, often private or real-time, data.

* LLMs are trained on enormous bodies of data but they aren’t trained on your data. RAG solves this problem by adding your data to the data LLMs already have access to.

* If you want to build AI applications that can reason about private data or data introduced after a model’s cutoff date, you need to augment the knowledge of the model with the specific information it needs. The process of bringing the appropriate information and inserting it into the model prompt is known as Retrieval Augmented Generation (RAG).

* In RAG, your data is loaded and prepared for queries or “indexed”. User queries act on the index, which filters your data down to the most relevant context. This context and your query then go to the LLM along with a prompt, and the LLM responds.

**Stages within RAG**

* There are five key stages within RAG, which in turn will be a part of any larger application you build. These are:

![rag](img/RAGadvanced.png)

* Self Query Retrival

* Parent Document Retriver

* Hybrid Search BM25 & Ensembles

* Contextual Compressors & Filters

* Hypothetical Document

* RAG Fusion

#### **Self Query Retrival**

* In self-query retrieval, the model generates a query based on the input prompt and then retrieves relevant documents using this query. Essentially, the model asks itself a question and retrieves information to answer it.

* This approach can be useful when the input prompt is incomplete or ambiguous. By generating a query, the model can focus its retrieval on specific aspects of the prompt.

![Self Query Retrival](img/SelfQuerying.jpeg)

**EXAMPLE**

```python
!pip -q install langchain huggingface_hub openai google-search-results tiktoken chromadb lark
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"
## Self-querying Retriever

from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings()
## Example data with metadata attached
docs = [
    Document(
        page_content="Complex, layered, rich red with dark fruit flavors",
        metadata={"name":"Opus One", "year": 2018, "rating": 96, "grape": "Cabernet Sauvignon", "color":"red", "country":"USA"},
    ),
    Document(
        page_content="Luxurious, sweet wine with flavors of honey, apricot, and peach",
        metadata={"name":"Château d'Yquem", "year": 2015, "rating": 98, "grape": "Sémillon", "color":"white", "country":"France"},
    ),
    Document(
        page_content="Full-bodied red with notes of black fruit and spice",
        metadata={"name":"Penfolds Grange", "year": 2017, "rating": 97, "grape": "Shiraz", "color":"red", "country":"Australia"},
    ),
    Document(
        page_content="Elegant, balanced red with herbal and berry nuances",
        metadata={"name":"Sassicaia", "year": 2016, "rating": 95, "grape": "Cabernet Franc", "color":"red", "country":"Italy"},
    ),
    Document(
        page_content="Highly sought-after Pinot Noir with red fruit and earthy notes",
        metadata={"name":"Domaine de la Romanée-Conti", "year": 2018, "rating": 100, "grape": "Pinot Noir", "color":"red", "country":"France"},
    ),
    Document(
        page_content="Crisp white with tropical fruit and citrus flavors",
        metadata={"name":"Cloudy Bay", "year": 2021, "rating": 92, "grape": "Sauvignon Blanc", "color":"white", "country":"New Zealand"},
    ),
    Document(
        page_content="Rich, complex Champagne with notes of brioche and citrus",
        metadata={"name":"Krug Grande Cuvée", "year": 2010, "rating": 93, "grape": "Chardonnay blend", "color":"sparkling", "country":"New Zealand"},
    ),
    Document(
        page_content="Intense, dark fruit flavors with hints of chocolate",
        metadata={"name":"Caymus Special Selection", "year": 2018, "rating": 96, "grape": "Cabernet Sauvignon", "color":"red", "country":"USA"},
    ),
    Document(
        page_content="Exotic, aromatic white with stone fruit and floral notes",
        metadata={"name":"Jermann Vintage Tunina", "year": 2020, "rating": 91, "grape": "Sauvignon Blanc blend", "color":"white", "country":"Italy"},
    ),
]
vectorstore = Chroma.from_documents(docs, embeddings)
## Creating our self-querying retriever
from langchain.llms import HuggingFaceHub
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="grape",
        description="The grape used to make the wine",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="name",
        description="The name of the wine",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="color",
        description="The color of the wine",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="year",
        description="The year the wine was released",
        type="integer",
    ),
    AttributeInfo(
        name="country",
        description="The name of the country the wine comes from",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="The Robert Parker rating for the wine 0-100", type="integer" #float
    ),
]
document_content_description = "Brief description of the wine"

llm = HuggingFaceHub(
    repo_id="google/flan-t5-base", 
    model_kwargs={"temperature": 0.5, "max_length": 512}
)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True
)

# This example only specifies a relevant query
retriever.get_relevant_documents("What are some red wines")

retriever.get_relevant_documents("I want a wine that has fruity nodes")

# This example specifies a query and a filter
retriever.get_relevant_documents("I want a wine that has fruity nodes and has a rating above 97")
retriever.get_relevant_documents("What wines come from Italy?")

# This example specifies a query and composite filter
retriever.get_relevant_documents("What's a wine after 2015 but before 2020 that's all earthy")

## Filter K

retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    enable_limit=True,
    verbose=True,
)

# This example only specifies a relevant query - k= 2
retriever.get_relevant_documents("what are two that have a rating above 97")
retriever.get_relevant_documents("what are two wines that come from australia or New zealand")
```

1. **Installation of Required Packages:** The first line installs the necessary packages using pip.

2. **Setting Hugging Face API Token:** An environment variable for the Hugging Face API token is set. This token is used for accessing models and resources on the Hugging Face model hub.

3. **Creating Example Data:** Example wine data with associated metadata is created. Each wine document consists of a page content (description of the wine) and metadata fields such as name, year, rating, grape variety, color, and country of origin.

4. **Embeddings and Vector Store:** HuggingFace embeddings are initialized. The `Chroma` vector store is then created from the documents and embeddings. This vector store enables efficient similarity search based on semantic embeddings.

5. **Defining Metadata Attributes:** Metadata attributes are defined using `AttributeInfo`. These attributes provide information about the metadata fields such as name, description, and data type.

6. **Initializing Hugging Face LLM (Large Language Model):** The Hugging Face model `google/flan-t5-base` is initialized as the Large Language Model (LLM) using the` HuggingFaceHub` class. This model will be used for generating queries based on user input.

7. **Creating Self-Querying Retriever:** The self-querying retriever is initialized using the `SelfQueryRetriever.from_llm` method. This retriever combines the LLM with the vector store and metadata information to retrieve relevant documents based on user queries.

8. **Querying the Retriever:** Various example queries are executed using the retriever's `get_relevant_documents` method. These queries demonstrate the system's ability to find relevant documents based on user-provided criteria such as wine characteristics, ratings, and country of origin.

9. **Filtering with Limit (K):** Another self-querying retriever is initialized with the option to enable a limit (`k`) on the number of retrieved documents. Example queries with a limit of two documents are then executed to demonstrate the filtering capability.

---

#### **Parent Document Retriver**

* A The parent document retriever aims to retrieve a broader context or a “parent” document related to the input prompt. It helps the model understand the context in which the prompt occurs.

* For example, if the prompt refers to a specific event, the parent document retriever could retrieve a longer article or document that provides background information about that event.

![Parent Document Retriver](img/ParentDocumentRetriver.png)

**1. Original Documents:** Start with the full set of documents that contain the information you want to query.

**2. Splitting Documents:** Split the original documents into smaller, more manageable parts called "parent chunks." These chunks should be of a decent size to maintain some context but not too large to become overly general.

**3. Creating Child Documents:** Each parent chunk is further split into smaller segments called "child documents." These child documents contain portions of the parent chunk, making them more specific and detailed.

**4. Embedding Child Documents:** Generate embeddings for each child document. Embeddings are vector representations of the documents that capture their semantic content. By having more specific child documents, the embeddings are also more specific and less likely to be diluted with irrelevant information.

**5. Query and Retrieval:**

* When a question is posed, it is converted into an embedding.
* This question embedding is then used to find the most relevant child document embeddings.

**6.Returning Parent Documents:** Instead of returning only the child document that matches the query embedding, the corresponding parent document is returned. This parent document provides a broader context, giving the language model more information to generate a comprehensive and accurate response.

![Parent Document Retriver](img/pdr.png)

**EXAMPLE**

```python
# Import necessary modules
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever

# For text splitting and document loading
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.document_loaders import TextLoader

# For embeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings

# Define the model name and embedding settings
model_name = "BAAI/bge-small-en-v1.5"
encode_kwargs = {'normalize_embeddings': True}  # Normalize to compute cosine similarity

# Initialize the BGE embeddings
bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs=encode_kwargs
)

# Load documents from text files
loaders = [
    TextLoader("blog_posts/blog.langchain.dev_announcing-langsmith_.txt", encoding='utf-8'),
    TextLoader('blog_posts/blog.langchain.dev_benchmarking-question-answering-over-csv-data_.txt', encoding='utf-8')
]

# Aggregate all loaded documents
docs = []
for loader in loaders:
    docs.extend(loader.load())

# Check the number of loaded documents
print(f"Number of loaded documents: {len(docs)}")

# Initialize a text splitter for creating small chunks
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# Set up the Chroma vector store for indexing child chunks
vectorstore = Chroma(
    collection_name="full_documents",
    embedding_function=bge_embeddings
)

# Initialize the in-memory storage for parent documents
store = InMemoryStore()

# Set up the ParentDocumentRetriever for the smaller chunks
full_doc_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)

# Add documents to the retriever
full_doc_retriever.add_documents(docs, ids=None)

# List stored keys to verify documents are added
print(f"Stored keys: {list(store.yield_keys())}")

# Perform a similarity search on the child chunks
sub_docs = vectorstore.similarity_search("what is langsmith", k=2)
print(f"Number of sub-documents retrieved: {len(sub_docs)}")
print(sub_docs[0].page_content)

# Retrieve full documents based on the query
retrieved_docs = full_doc_retriever.get_relevant_documents("what is langsmith")
print(f"Number of retrieved documents: {len(retrieved_docs)}")
print(retrieved_docs[0].page_content)

# Retrieving larger chunks
# Initialize a text splitter for creating larger parent chunks
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

# Reinitialize the text splitter for smaller child chunks (as before)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# Set up another Chroma vector store for the split parent documents
vectorstore = Chroma(
    collection_name="split_parents",
    embedding_function=bge_embeddings
)

# Reinitialize the in-memory storage
store = InMemoryStore()

# Set up the ParentDocumentRetriever for the larger parent chunks
big_chunks_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# Add documents to the new retriever
big_chunks_retriever.add_documents(docs)

# Verify the added documents
print(f"Stored keys in new retriever: {len(list(store.yield_keys()))}")

# Perform a similarity search on the larger parent chunks
sub_docs = vectorstore.similarity_search("what is langsmith")
print(f"Number of sub-documents retrieved: {len(sub_docs)}")
print(sub_docs[0].page_content)

# Retrieve full parent documents based on the query
retrieved_docs = big_chunks_retriever.get_relevant_documents("what is langsmith")
print(f"Number of retrieved documents: {len(retrieved_docs)}")
print(retrieved_docs[0].page_content)
print(retrieved_docs[1].page_content)

# Example of using RetrievalQA for querying the documents
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

# Initialize RetrievalQA with the larger chunk retriever
qa = RetrievalQA.from_chain_type(
    llm=HuggingFaceHub(),
    chain_type="stuff",  # Ensure this matches your actual setup
    retriever=big_chunks_retriever
)

# Perform a query
query = "What is Langsmith?"
answer = qa.run(query)
print(f"Answer: {answer}")
```

**Explanation:**

1. **Import Necessary Modules:** Import all necessary modules from LangChain, Chroma, and Hugging Face.

2. **Model and Embeddings Setup:** Define the model name for BGE embeddings and initialize it with appropriate settings.

3. **Document Loading:** Load documents from text files using `TextLoader` and aggregate them into a list.

4. **Small Chunk Splitting:** Use `RecursiveCharacterTextSplitter` to split documents into small chunks (400 characters).

5. **Chroma Vector Store Setup:** Initialize the Chroma vector store to index the small chunks.

6. **In-Memory Storage Setup:** Set up an in-memory store for parent documents.

7. **ParentDocumentRetriever for Small Chunks:** Set up `ParentDocumentRetriever` to handle small chunks and add the documents to it.

8. **Verify Document Addition:** List the keys in the storage to verify the documents have been added.

9. **Perform Similarity Search:** Use the Chroma vector store to perform a similarity search on the small chunks.

10. **Retrieve Full Documents:** Retrieve full documents based on the query using the `ParentDocumentRetriever`.

11. **Large Chunk Splitting:** Reinitialize text splitters for larger parent chunks (2000 characters) and smaller child chunks (400 characters).

12. **Set Up New Chroma Vector Store:** Initialize another Chroma vector store for the larger parent chunks.

13. **Reinitialize In-Memory Storage:** Reinitialize the in-memory storage.

14. **ParentDocumentRetriever for Large Chunks:** Set up another `ParentDocumentRetriever` for the larger chunks and add the documents to it.

15. **Verify New Document Addition:** Verify the documents added to the new retriever.

16. **Perform Similarity Search on Large Chunks:** Perform a similarity search on the larger parent chunks.

17. **Retrieve Full Parent Documents:** Retrieve full parent documents based on the query using the new `ParentDocumentRetriever`.

18. **RetrievalQA Example:** Set up `RetrievalQA` to query the documents and perform a sample query.

---

#### **Hybrid Search BM25 & Ensembles**

* Hybrid search integrates the strengths of both keyword-based and semantic-based search methods to enhance retrieval performance. Here, we will define and explain the key concepts involved, particularly focusing on BM25 and how it works within a hybrid search framework.

![Hybrid Search](img/HybridSearchBM25Ensembles.webp)

**BM25 Overview**

* BM25 (Best Matching 25) is a ranking function used in information retrieval and search engines to rank documents based on the relevance to a given query. It is a type of bag-of-words model and an improved version of the traditional TF-IDF (Term Frequency-Inverse Document Frequency) model. BM25 has been widely used since the 1970s and 1980s and remains a strong baseline for text retrieval tasks.

**What it does:** It looks at how often your search words appear in a document and considers the document’s length to provide the most relevant results.

**Why it’s useful:** It’s perfect for sorting through huge collections of documents, like a digital library, without bias towards longer documents or overused words.

**Key Components of BM25:**

  1. **Term Frequency (TF):** This counts how many times your search terms appear in a document.

  2. **Inverse Document Frequency (IDF):** This gives more importance to rare terms, making sure common words don’t dominate.

  3. **Document Length Normalization:** This ensures longer documents don’t unfairly dominate the results.

  4. **Query Term Saturation:** This stops excessively repeated terms from skewing the results.

BM25 creates sparse vectors where each dimension corresponds to a term, and the values represent term weights based on the TF-IDF score.

![BM25](img/bm25A.png)

**When is BM25/ Keyword search Ideal?**

  1. **Large Document Collections:** Perfect for big databases where you need to sort through lots of information.
  
  2. **Preventing Bias:** Great for balancing term frequency and document length.

  3. **General Information Retrieval:** Useful in various search scenarios, offering a mix of simplicity and effectiveness.

**Hybrid Search**

Hybrid search combines the traditional keyword search (like BM25) with vector-based search methods that leverage embeddings from deep learning models. This combination aims to utilize the precise matching of keyword search and the contextual understanding of vector search.

**Components of Hybrid Search:**

1. **Keyword Search (BM25):**

    1. **Precision:** BM25 is excellent at exact term matching and ranking documents based on keyword frequency and distribution.

    2. **Simplicity:** It uses straightforward counting mechanisms, making it efficient and interpretable.

    3. **Sparse Representation:** Documents and queries are represented as sparse vectors, focusing on specific terms.

2. **Vector Search (Embeddings):**

3. **Contextual Understanding:** Uses embeddings to capture the semantic meaning of words and phrases, allowing for understanding synonyms and related terms.

4. **Dense Representation:** Documents and queries are represented as dense vectors in a high-dimensional space, capturing nuanced meanings.

**How Hybrid Search Works:**

1. **Initial Query Processing:** The user's query is processed in two ways—using BM25 for keyword-based retrieval and using an embedding model for semantic retrieval.

2. **Retrieval Stage:**

   1. **BM25 Retrieval:** The query is matched against documents using BM25, retrieving a set of documents based on exact term matches and their relevance scores.

   2. **Vector Retrieval:** The query is embedded into a vector, and similar document vectors are retrieved based on cosine similarity or other distance metrics.

**EXAMPLE**

```python
# Install Required Libraries:
# !pip -q install langchain huggingface_hub openai google-search-results tiktoken chromadb rank_bm25 faiss-cpu

import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "Enter your key here"
# Import Necessary Modules:
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
# Prepare Document List:
doc_list = [
    "I like apples",
    "I like oranges",
    "Apples and oranges are fruits",
    "I like computers by Apple",
    "I love fruit juice"
]
# Initialize BM25 Retriever:
bm25_retriever = BM25Retriever.from_texts(doc_list)
bm25_retriever.k = 4  # Set the number of documents to retrieve
# Retrieve documents using BM25:
bm25_docs_apple = bm25_retriever.get_relevant_documents("Apple")
bm25_docs_green_fruit = bm25_retriever.get_relevant_documents("a green fruit")
# Initialize FAISS Retriever:
embeddings = HuggingFaceEmbeddings()
faiss_vectorstore = FAISS.from_texts(doc_list, embeddings)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 4})
# Retrieve documents using FAISS:
faiss_docs_green_fruit = faiss_retriever.get_relevant_documents("A green fruit")
# Combine Retrievers Using EnsembleRetriever:
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
# Perform Searches Using Ensemble Retriever:
docs_green_fruit = ensemble_retriever.get_relevant_documents("A green fruit")
docs_apple_phones = ensemble_retriever.get_relevant_documents("Apple Phones")
```

**1. Install Required Libraries:** Ensure you have all the necessary libraries installed. You have already done this step.

**2. Import Necessary Modules:** Import the modules required for setting up BM25 and FAISS retrievers.

**3. Prepare Document List:** Create a list of documents that we will use for retrieval.

**4. Initialize BM25 Retriever:** Set up the BM25 retriever which is a sparse retriever.

**5. Initialize FAISS Retriever:** Set up the FAISS retriever which is a dense retriever.

**6. Combine Retrievers Using EnsembleRetriever:** Use the EnsembleRetriever to combine both retrievers for hybrid search.

**7. Perform Searches:** Perform searches using the ensemble retriever and observe the results.

**BM25 Retriever:** It uses a bag-of-words approach to retrieve documents based on the occurrence of query terms.

**FAISS Retriever:** It utilizes dense vector embeddings to retrieve documents based on the semantic similarity to the query.

**Ensemble Retriever:** Combines the results of both BM25 and FAISS retrievers to provide a more comprehensive set of relevant documents.

---

#### **Contextual Compressors & Filters**

* Contextual Compressors and Filters are tools used in information retrieval systems to refine and extract relevant information from retrieved documents based on the context of a given query. These tools aim to enhance the efficiency and effectiveness of retrieval by presenting only the most useful and pertinent information to downstream processing components.

**Components of Contextual Compressors and Filters:**

1. **Base Retriever:**

* Initially retrieves a set of documents or pieces of information relevant to the query.

2. **Document Compressors and Filters:**

* Process the retrieved documents to extract and refine the information that is most useful for answering the query.

* These tools operate based on the context of the query and the content of the documents, aiming to filter out irrelevant or extraneous information.

* Examples of operations performed by compressors and filters include:

  1. Removing irrelevant sections of documents.
  2. Extracting key information from documents.
  3. Applying language models or other machine learning techniques to identify and select relevant content.

**Workflow of Contextual Compressors and Filters:**

1. **Document Retrieval:**

* Begin with a base retriever that fetches a collection of documents relevant to the query.

2. **Document Processing:**

* Contextual compressors and filters analyze the retrieved documents to identify and extract relevant information.
This may involve various operations, such as cleaning up the document content, removing noise, and extracting essential details.

3. **Refined Document Set:**

* The processed documents form a refined set of information, containing only the most relevant and useful content for answering the query.
This refined set serves as input for further processing or analysis.

4. **Evaluation and Feedback:**

* Optionally, the effectiveness of the compressors and filters can be evaluated by comparing the refined document set with the original retrieved documents.

**EXAMPLE**

```python
import os
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.llms import HuggingFaceHub
from langchain.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter

# Setting up Hugging Face BGE Embeddings
model_name = "BAAI/bge-small-en-v1.5"
encode_kwargs = {'normalize_embeddings': True}
bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs=encode_kwargs
)

# Loading and Splitting Text Documents
loaders = [
    TextLoader("blog_posts/blog.langchain.dev_announcing-langsmith_.txt", encoding='utf-8'),
    TextLoader('blog_posts/blog.langchain.dev_benchmarking-question-answering-over-csv-data_.txt', encoding='utf-8')
]
docs = []
for loader in loaders:
    docs.extend(loader.load())
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(docs)

# Creating a Document Retriever with FAISS
retriever = FAISS.from_documents(texts, bge_embeddings).as_retriever()

# Performing Document Retrieval
docs = retriever.get_relevant_documents("What is LangSmith?")

# Adding Contextual Compression
llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.5, "max_length": 512})
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

# Retrieving Compressed Documents
compressed_docs = compression_retriever.get_relevant_documents("What is LangSmith?")

# Using LLMChainFilter for Compression
_filter = LLMChainFilter.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(base_compressor=_filter, base_retriever=retriever)
compressed_docs = compression_retriever.get_relevant_documents("What is LangSmith?")
```

1. **Importing necessary libraries:**

**`os`:** To interact with the operating system.

**`langchain`:** A library for natural language processing tasks.

**`HuggingFaceEmbeddings`:** Embeddings from the Hugging Face library.

**`TextLoader`, `RecursiveCharacterTextSplitter`, `InMemoryStore`:**Components for loading and processing text data.

**`HuggingFaceBgeEmbeddings`:`**Embeddings specifically designed for the LangChain library.

**`FAISS`:** A library for efficient similarity search and clustering of dense vectors.

**`Chroma`:** A component for representing text data.

Other specific modules and components from `langchain`.

2. **Setting up Hugging Face BGE (Big Green Egg) embeddings:**

* Specifying the model name and any additional arguments.

* Creating an instance of `HuggingFaceBgeEmbeddings`.

3. **Loading and splitting text documents:**

* Creating instances of `TextLoader` for each text file.

* Loading documents from text files.

* Splitting the documents into smaller chunks using `CharacterTextSplitter`.

4. Defining a helper function `pretty_print_docs` to print out the loaded documents nicely.

5. **Using the FAISS library to create a document retriever:**

* Creating a FAISS index from the documents.

* Using the BGE embeddings for vector representation.

* Getting relevant documents based on a query ("What is LangSmith?").

6. **Introducing contextual compression using an LLMChainExtractor:**

* Creating an instance of HuggingFaceHub for a language model.

* Creating an LLMChainExtractor from the language model.

* Using the extractor as a compressor in a ContextualCompressionRetriever.

7. Retrieving relevant documents after compression based on the same query.

8. **Exploring another compression technique using LLMChainFilter:**

* Creating an LLMChainFilter from the same language model.

* Using the filter as a compressor in a new ContextualCompressionRetriever.

* Retrieving relevant documents again based on the query.

---

#### **Hypothetical Document**

An alternative method involves prompting an LLM to formulate a question for every chunk, embedding these questions into vectors. During runtime, a query search is conducted against this index of question vectors, replacing the chunk vectors with question vectors in our index. Upon retrieval, the original text chunks are routed and provided as context for the LLM to generate an answer. The quality of the search will be better as there is higher semantic similarity between the query and the embedded hypothetical question.

> This method is the reverse of another approach called HyDE where the LLM generates a hypothetical response for the query. The response vector in conjunction with the query vector enhances search quality.

![Hypothetical Document](img/HypotheticalDocumentEmbeddings.jpg)

**EXAMPLE**

```python
!pip -q install langchain huggingface_hub openai chromadb tiktoken faiss-cpu
!pip install sentence_transformers
!pip -q install -U FlagEmbedding

import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"

from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate

from langchain.document_loaders import TextLoader
import langchain
from langchain.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-small-en-v1.5"
encode_kwargs = {'normalize_embeddings': True}

bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs=encode_kwargs
)

llm = HuggingFaceHub()

embeddings = HypotheticalDocumentEmbedder.from_llm(llm,
                                                   bge_embeddings,
                                                   prompt_key="web_search"
                                                   )

embeddings.llm_chain.prompt

langchain.debug = True

result = embeddings.embed_query("What items does McDonalds make?")

multi_llm = HuggingFaceHub(repo_id="google/flan-t5-base", huggingfacehub_api_token="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")

def generate_best_response(prompt, n=4, best_of=4):
    responses = [multi_llm(prompt) for _ in range(n)]
    best_response = max(responses, key=lambda response: len(response))
    return best_response

embeddings = HypotheticalDocumentEmbedder.from_llm(
    multi_llm, bge_embeddings, "web_search"
)

result = embeddings.embed_query("What is McDonalds best selling item?")

prompt_template = """Please answer the user's question as a single food item
Question: {question}
Answer:"""

prompt = PromptTemplate(input_variables=["question"], template=prompt_template)

llm_chain = LLMChain(llm=llm, prompt=prompt)

embeddings = HypotheticalDocumentEmbedder(
    llm_chain=llm_chain,
    base_embeddings=bge_embeddings
)

result = embeddings.embed_query(
    "What is is McDonalds best selling item?"
)

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

loaders = [
    TextLoader("blog_posts/blog.langchain.dev_announcing-langsmith_.txt", encoding='utf-8'),
    TextLoader('blog_posts/blog.langchain.dev_benchmarking-question-answering-over-csv-data_.txt', encoding='utf-8'),
    TextLoader('blog_posts/blog.langchain.dev_chat-loaders-finetune-a-chatmodel-in-your-voice_.txt', encoding='utf-8')
]

docs = []

for loader in loaders:
    try:
        docs.extend(loader.load())

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

prompt_template = """Please answer the user's question as related to Large Language Models
Question: {question}
Answer:"""

prompt = PromptTemplate(input_variables=["question"], template=prompt_template)

llm_chain = LLMChain(llm=llm, prompt=prompt)

embeddings = HypotheticalDocumentEmbedder(
    llm_chain=llm_chain,
    base_embeddings=bge_embeddings
)

docsearch = Chroma.from_documents(texts, embeddings)

query = "What are chat loaders?"
docs = docsearch.similarity_search(query)

print(docs[0].page_content)

```

* It installs required Python packages for natural language processing and related tasks.

* Sets up environment variables for authentication with Hugging Face Hub.

* Imports necessary modules and classes from the LangChain library.

* Configures Hugging Face BGE embeddings and initializes a Hugging Face language model (LLM).

* Sets up embeddings for hypothetical documents using the initialized LLM and BGE embeddings.

* Defines a function to generate the best response given a prompt.

* Defines prompt templates for generating responses to user questions.

* Initializes a Language Model Chain with the LLM and prompt templates.

* Loads and splits text documents for further processing.

* Initializes a Chroma object for searching documents based on embeddings.

* Performs a similarity search for documents relevant to a specific query.

* Prints the content of the most relevant document returned by the similarity search.

---

#### **RAG Fusion**

* RAG Fusion is an approach aimed at enhancing the Retrieval-Augmented Generation (RAG) model by addressing the gap between what users explicitly ask and what they actually intend to ask. This technique is particularly valuable in scenarios where users input vague or broad queries, but desire comprehensive and diverse responses.

* The key components of RAG Fusion include:

  1. Query Duplication with a Twist: RAG Fusion starts by rewriting the user's input query into multiple similar queries. Each of these modified queries is then processed independently through a vector search to retrieve different sets of results.

  2. Vector Search and Result Ranking: The modified queries undergo vector searches, retrieving distinct sets of relevant information. These results are subsequently ranked using reciprocal rank fusion, which ensures that the most pertinent information is prioritized across multiple queries.

  3. Generative Output Integration: The re-ranked outputs from the vector searches are treated as contextual inputs and combined with the original query and prompt. This integrated information is then fed into a generative model, such as a large language model (LLM), to formulate a comprehensive response.

**EXAMPLE**

```python
import os
import requests
import zipfile
from io import BytesIO
import textwrap
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chat_models import ChatGooglePalm
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.load import dumps, loads
from operator import itemgetter

# Function to download and extract zip files
def download_and_extract_zip(url, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download file: {url}")

    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(target_folder)

    print(f"Files extracted to {target_folder}")

# Set environment variable for HuggingFaceHub API token
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"

# Download and extract text files and Chroma database
text_files_url = "https://www.dropbox.com/scl/fi/av3nw07o5mo29cjokyp41/singapore_text_files_languages.zip?rlkey=xqdy5f1modtbnrzzga9024jyw&dl=1"
chroma_db_url = 'https://www.dropbox.com/scl/fi/3kep8mo77h642kvpum2p7/singapore_chroma_db.zip?rlkey=4ry4rtmeqdcixjzxobtmaajzo&dl=1'
text_files_folder = "singapore_text"
chroma_db_folder = "chroma_db"

download_and_extract_zip(text_files_url, text_files_folder)
download_and_extract_zip(chroma_db_url, '.')

# Load documents
loader = DirectoryLoader('singapore_text/Textfiles3/English/', glob="*.txt", show_progress=True)
docs = loader.load()

# Concatenate and split text
raw_text = ''.join([doc.page_content for doc in docs if doc.page_content])
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, length_function=len, is_separator_regex=False)
texts = text_splitter.split_text(raw_text)

# Load embeddings and vector database
model_name = "BAAI/bge-small-en-v1.5"
encode_kwargs = {'normalize_embeddings': True}
bge_embeddings = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'}, encode_kwargs=encode_kwargs)

db = Chroma(persist_directory="./chroma_db", embedding_function=bge_embeddings)

# Retriever and Chat Model setup
retriever = db.as_retriever(k=5)
model = ChatGooglePalm()

# Prompt template for RAG
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Function to generate multiple queries
generate_queries_prompt = ChatPromptTemplate(input_variables=['question'], messages=[
    SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='You are a helpful assistant that generates multiple search queries based on a single input query.')),
    HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['question'], template='Generate multiple search queries related to: {question} \n OUTPUT (4 queries):'))
])
generate_queries = generate_queries_prompt | ChatGooglePalm(temperature=0) | StrOutputParser() | (lambda x: x.split("\n"))

# Reciprocal Rank Fusion function
def reciprocal_rank_fusion(results, k=60):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    reranked_results = [(loads(doc), score) for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
    return reranked_results

# RAG Fusion chain
ragfusion_chain = generate_queries | retriever.map() | reciprocal_rank_fusion

# Full RAG Fusion chain with prompt
full_rag_fusion_chain = (
    {
        "context": ragfusion_chain,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)

# Query and get response
query = "Tell me about Universal Studios Singapore?"
response = full_rag_fusion_chain.invoke({"question": query, "original_query": query})

# Wrap and print the response
def wrap_text(text, width=90):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    return '\n'.join(wrapped_lines)

print(wrap_text(response))

```

1. **Download and Extraction Functions:**

* **download_and_extract_zip:** This function downloads a zip file from a given URL and extracts its contents to a specified target folder.

2. **Setting Environment Variables:**

* Sets the environment variable `HUGGINGFACEHUB_API_TOKEN` with a Hugging Face Hub API token.

3. **Downloading Text Files and Chroma Database:**

* Downloads and extracts text files and a Chroma database from Dropbox URLs.

4. **Loading Documents:**

* Loads text documents from the extracted text files.

5. **Text Preprocessing:**

* Concatenates the text from all documents and splits it into smaller chunks.

6. **Loading Embeddings and Vector Database:**

* Loads pre-trained embeddings using a Hugging Face model and creates a vector database using Chroma.

7. **Retriever and Chat Model Setup:**

* Configures a retriever using the Chroma vector database and sets up a conversational AI model using Google PaLM.

8. **Prompt Template for RAG (Retrieval-Augmented Generation):**

* Defines a template for prompts used in the RAG fusion chain.

9. **Function to Generate Multiple Queries:**

* Defines a function that generates multiple search queries based on a single input query using the Google PaLM model.

10. **Reciprocal Rank Fusion Function:**

* Defines a function to perform reciprocal rank fusion on retrieval results.

11. **RAG Fusion Chain:**

* Constructs a fusion chain that generates multiple search queries, retrieves documents, and performs fusion using the reciprocal rank fusion function.

12. **Full RAG Fusion Chain with Prompt:**

* Combines the RAG fusion chain with a prompt for generating responses to a user query.

13. **Querying and Getting Response:**

* Executes the full RAG fusion chain with a specific query and obtains the response.

14. **Wrapping and Printing the Response:**

* Wraps the response text to a specified width and prints it.

---

### **Transformers**

Transformers are a type of deep learning model architecture that has revolutionized the field of natural language processing (NLP). Introduced by Vaswani et al. in the 2017 paper "Attention Is All You Need," the Transformer architecture primarily relies on self-attention mechanisms to process input data in parallel, which contrasts with the sequential processing of earlier recurrent neural networks (RNNs). This enables the model to understand the context and relevance of each word in a sentence more effectively. Key components of Transformers include tokenization, positional encoding, self-attention mechanisms, multi-headed attention, encoder and decoder structures, feed-forward neural networks, normalization, and residual connections.

![Transformers](img/TransformersArchitecture.png)

**Background on Neural Networks**

Before diving into transformers, it's essential to understand the context of neural networks. Neural networks are models designed to analyze complex data types like images, videos, audio, and text. Different types of neural networks are optimized for different kinds of data. For instance:

![neural_networks](img/neuralnetwork.png)

* Convolutional Neural Networks (CNNs) are typically used for image processing, mimicking how the human brain processes visual information.

![cnn](img/nn.png)

**The Problem with Recurrent Neural Networks (RNNs)**

Before transformers, Recurrent Neural Networks (RNNs) were the go-to models for language tasks. RNNs process text sequentially, word by word, which helps maintain the order of words—a crucial factor in understanding language. However, RNNs had significant drawbacks:

* **Difficulty handling long sequences:** RNNs struggled with long texts, often forgetting earlier parts of the sequence by the time they processed the end.

* **Training challenges:** Their sequential nature made them hard to parallelize, resulting in slow training times and limited scalability.

**The Rise of Transformers**

Transformers, introduced in 2017 by researchers at Google and the University of Toronto, addressed these issues and changed the landscape of NLP. Unlike RNNs, transformers can be efficiently parallelized, enabling the training of much larger models. For example, GPT-3, a well-known transformer model, was trained on approximately 45 terabytes of text data, including nearly the entire public web.

**Why Should You Use Transformers?**

1. **State-of-the-Art Performance:**

    * Achieve high performance on various tasks such as language translation, text summarization, question answering, image classification, and speech recognition.

2. **Ease of Use:**

    * Low Barrier to Entry: Suitable for educators and practitioners with minimal deep learning experience.
    
    * Simple Abstractions: Involves learning just three main classes, making it accessible.

3. **Unified API:**

    * Provides access to a wide array of pretrained models with a unified API, facilitating easy implementation and experimentation.

4. **Cost and Efficiency:**

    * **Reduced Compute Costs:** Allows researchers to share pretrained models, reducing the need for expensive training from scratch.

    * **Lower Carbon Footprint:** Efficient use of computational resources, contributing to more sustainable AI practices.

5. **Flexibility Across Frameworks:**

    * **Interoperability:** Models can be easily moved between TensorFlow 2.0, PyTorch, and JAX frameworks.

    * **Optimized Training:** Train state-of-the-art models with minimal code.

6. **Customization:**

    * **Reproducible Results:** Examples provided to help reproduce results published by original authors.

    * **Exposed Model Internals:** Allows for deep customization and quick experimentation.

7. **Comprehensive Model Library:**

    * **Diverse Architectures:** Access to over 400,000 pretrained models across various domains and tasks.

**Why Shouldn't You Use Transformers?**

1. **Not a Modular Toolbox:**

    * **Lack of Generalization:** The library is not designed as a modular toolbox for building neural nets from scratch. It’s optimized for the specific models provided.

2. **Specialized Training API:**

    * **Model-Specific Training:** The training API is tailored for the models within the library and may not be suitable for generic machine learning tasks. For more generalized training loops, another library like Accelerate might be more appropriate.

**How do transformers work?**

![transformers](img/Transformers.png)

1. **Positional Encodings:**

* Transformers use positional encodings to store information about the order of words in a sentence. Each word is tagged with a positional number before being fed into the network, allowing the model to learn word order from the data itself.

![Positional Encodings](img/Transformers2.png)

2. **Attention Mechanism:**

* The attention mechanism enables the model to focus on different parts of the input text when making predictions. This is crucial for tasks like translation, where the model needs to consider the entire sentence to generate the correct output.
  
* A visualization from the original transformer paper shows how the model attends to different words in the input sentence when predicting the output words.

![Positional Encodings](img/Transformers1.png)

3. **Self-Attention:**

* Self-attention is a specific type of attention where the model considers the input text itself. This helps the model understand the context and disambiguate words with multiple meanings based on their surrounding words. For example, in the sentences "Server, can I have the check?" and "Looks like I just crashed the server," self-attention helps distinguish between a human server and a computer server.

![Self-Attention](img/Self-attention.png)

---

## References

1. https://github.com/sudarshan-koirala/youtube-stuffs/blob/main/langchain/LangChain_Components.ipynb

2. https://github.com/PradipNichite/Youtube-Tutorials/blob/main/Youtube_Course_Sentence_Transformers.ipynb

3. https://www.youtube.com/watch?v=jbFHpJhkga8&list=PLz-qytj7eIWVd1a5SsQ1dzOjVDHdgC1Ck

4. https://www.youtube.com/watch?v=nAmC7SoVLd8list=PLeo1K3hjS3uu0N_0W6giDXzZIcB07Ng_F

5. https://www.youtube.com/watch?v=mBJqGAHoam4

6. https://langchain-cn.readthedocs.io/en/latest/modules/models/text_embedding/examples/huggingfacehub.html

7. https://www.youtube.com/watch?v=rh7aB-Pbxa4

8. https://www.youtube.com/watch?v=RC3JOzN2F1w

9. https://www.youtube.com/watch?v=SZorAJ4I-sA

10. https://www.youtube.com/watch?v=iUmL4p_N79I

11. https://www.promptingguide.ai/techniques

12. https://pub.towardsai.net/advanced-rag-techniques-an-illustrated-overview-04d193d8fec6

13. https://www.youtube.com/playlist?list=PL8motc6AQftn-X1HkaGG9KjmKtWImCKJS

**Transformers**

14. https://www.youtube.com/watch?v=nTlLAS7N7qE

15. https://www.youtube.com/watch?v=BjRVS2wTtcA

16. https://www.youtube.com/watch?v=BjRVS2wTtcA

17. https://www.youtube.com/watch?v=ZXiruGOCn9s