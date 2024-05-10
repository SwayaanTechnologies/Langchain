# Langchain

![logo](https://media.gettyimages.com/id/1801115823/photo/in-this-photo-illustration-the-langchain-logo-is-displayed.jpg?b=1&s=594x594&w=0&k=20&c=OpkcRRc6G8I_-jYYk4Tgu5gWVtgYilTypQ4naXcNJqU=)

## TABLE OF CONTENT

1. [**Introduction to Langchain**](#Introduction-to-Langchain)

2. [**Components**](#Components)

    * [**Schema**](#Schema)

    * [**Models**](#Models)

    * [**Prompts**](#Prompts)

    * [**Parsers**](#Parsers)

    * [**Indexes**](#Indexes)

        - [**Document Loading**](#Document-Loading)   

        - [**Documnet Splitting**](#Documnet-Splitting)  

        - [**Vectors and Embeddings**](#Vectors-and-Embeddings) 

        - [**Retrevial**](#retrevial)

    * [**Memory**](#Memory)

    * [**Chains**](#Chains)

    * [**Agents**](#Agents)

3. [**References**](#References)

---

## Introduction to Langchain

* LangChain is an open-source framework designed to simplify the creation of applications using large language models (LLMs). It provides a standard interface for chains, lots of integrations with other tools, and end-to-end chains for common applications. It allows AI developers to develop applications based on the combined Large Language Models (LLMs) such as GPT-4 with external sources of computation and data. This framework comes with a package for both Python and JavaScript.

* LangChain follows a general pipeline where a user asks a question to the language model where the vector representation of the question is used to do a similarity search in the vector database and the relevant information is fetched from the vector database and the response is later fed to the language model. further, the language model generates an answer or takes an action.

---

### Evolve 

* The journey of LangChain began as an ambitious project to overcome the limitations of early language models. Its evolution is marked by significant milestones that reflect the rapid advancement of AI and NLP technologies. Initially, language models were constrained by simplistic rule-based systems that lacked the ability to understand context or generate natural-sounding text. As machine learning and deep learning techniques matured, the foundation for LangChain was set.

* The advancements in transfer learning further propelled LangChain, making it possible to fine-tune models on specific datasets. This adaptability made LangChain a versatile tool for developers in various fields.

* The integration of modular components for specialized linguistic tasks expanded LangChain’s capabilities. Developers could extend LangChain’s functionality by adding or removing modules tailored to their needs, such as sentiment analysis, language translation, and more.

* Throughout its history, LangChain has placed a significant focus on context retention. Early language models struggled to maintain context over extended conversations, but LangChain introduced advanced memory mechanisms, allowing it to remember and reference past interactions, thereby creating more natural and engaging dialogues.

* Today, LangChain stands as a testament to the progress in AI conversational systems. With each update and refinement, it has become more sophisticated, more intuitive, and more capable of delivering experiences that closely mimic human interaction. It’s a story of continual improvement and innovation, with the promise of further advancements as the AI field evolves.

* LangChain’s ongoing development is driven by a community of researchers, developers, and enthusiasts who are relentlessly pushing the boundaries of what’s possible in AI. As we look back at its brief but impactful history, it is clear that LangChain is not just following the trends in AI development—it is setting them, paving the way for a future where conversational AI becomes an integral part of our daily lives. It’s exciting to think about what the future holds for LangChain and AI in general!

---

### Why do we need Langchain?
LangChain allows developers to create data-aware and agentic applications that can interact with their environment and leverage the power of large language models. Here are some use cases and examples of applications built with LangChain:

* **Autonomous agents:** LangChain can be used to create autonomous agents that can write code, run tests, and deploy applications using natural language commands. This is particularly useful for automating repetitive tasks and improving productivity.

* **Agent simulations:** LangChain can be used to simulate the behavior and interactions of multiple agents in a sandbox environment. This can be used to test the long-term memory and social skills of language models or explore how they react to different events or scenarios.

* **Personal assistants:** LangChain can be used to create personal assistants that can access and manipulate user data, remember user preferences and history, and perform various tasks such as booking flights, ordering food, or sending emails. This can greatly enhance the user experience and make interactions with technology more natural and intuitive.

* **Question answering:** LangChain can be used to create question-answering applications that can extract relevant information from text, images, audio, or video files, and provide concise and accurate answers to user queries. This can be particularly useful in fields like customer service, where quick and accurate responses are crucial.

---

## Components

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

7. [**Chains**](#Chains)

8. [**Agents**](#Agents)

![Components](img/companes.png)

---

### Schema

* The schema in LangChain can be defined using various techniques and languages, depending on the specific requirements and technologies used in the project. Commonly used schema definition languages include SQL (Structured Query Language), JSON (JavaScript Object Notation), and YAML (YAML Ain’t Markup Language).

* By defining a clear and consistent schema, LangChain ensures that data is organized and standardized, facilitating efficient data retrieval and manipulation. This is crucial for the performance and reliability of applications built with LangChain. It also ensures compatibility and interoperability between different components, making it easier for developers to build and manage their applications.

```python
import promptlayer
import os
os.environ["PROMPTLAYER_API_KEY"] = "<your-api-key>"
```

This section imports the promptlayer library and sets the environment variable PROMPTLAYER_API_KEY to your API key. This API key is required for using the PromptLayer service, which provides an interface for interacting with language models.

```python
#from langchain.schema import (HumanMessage,SystemMessage,AIMessage)
#from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.schema import (SystemMessage,HumanMessage,AIMessage)
```

These lines are commented out but suggest importing various components from the LangChain library, such as message schemas and chat models. However, it seems they are not used in the subsequent code.

```python
messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(
        content="What happens when an unstoppable force meets an immovable object?"
    ),
]
```

Here, a list of messages is created, consisting of a system message and a human message. The system message appears to set the context, while the human message poses a question.

```python
chat = PromptLayerChatOpenAI(pl_tags=["langchain"])
```

An instance of PromptLayerChatOpenAI is created, specifying the tags "langchain". This class likely facilitates interactions with the PromptLayer service using OpenAI's API.

```python
chat([
 SystemMessage(content="You are a helpful assistant that translates English to French."),
 HumanMessage(content="Translate this sentence from English to French. I love programming.")
])
```

This block of code initiates a chat interaction by providing a list of messages to the chat instance. The system message sets the context, and the human message poses a translation task from English to French.

```python
chat = PromptLayerChatOpenAI(temperature=0.1)
```

A new instance of `PromptLayerChatOpenAI` is created, this time setting the temperature parameter to 0.1. Temperature is a parameter that controls the randomness of the language model's responses during generation.

```python
# multiple sets of messages using .generate.
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
```

A list of multiple sets of messages is created, each containing a system message and a human message. These sets of messages are intended to be used in a batch for generating responses.

```python
result = chat.generate(batch_messages)
print(result)
```

The `generate` method of the chat instance is called with the batch messages as input. This method likely generates responses for each set of messages in the batch and returns the results. The results are then printed.

```python
chat([HumanMessage(content="Translate this sentence from English to tamil. I love programming.")])
```

Another human message is sent to the chat instance, this time requesting translation from English to Tamil.

### Models

models, such as GPT-4, are trained on vast amounts of text data and can generate human-like text based on the input they are given. They are the core of LangChain applications, enabling capabilities like natural language understanding and generation.

LangChain provides a standard interface for interacting with these models, abstracting away many of the complexities involved in working directly with LLMs. This makes it easier for developers to build applications that leverage the power of these models.

**Loading Environment Variable**

```python
import promptlayer
import os
os.environ["PROMPTLAYER_API_KEY"] = "<your-api-key>"
```

Here, the code imports the `promptlayer` module and sets the environment variable `PROMPTLAYER_API_KEY` to a specific API key. This API key is likely used for accessing a service that provides language model capabilities.

**Setting Model Variable**

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

This section determines which language model to use based on the current date. If the current date is before June 12, 2024, it sets the `llm_model` variable to "gpt-3.5-turbo". Otherwise, it sets it to "gpt-3.5-turbo-0301". This decision might be based on model updates or improvements.


**Defining Completion Function**

```python
def get_completion(prompt, model=llm_model):
    messages = [{"role": "user", "content": prompt}]
    response = PromptLayerChatOpenAI.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.1,
    )
    return response.choices[0].message["content"]
```

This function, `get_completion`, takes a prompt and an optional model name as input. It then uses the specified model to generate a completion for the provided prompt. The completion is generated with a low temperature parameter (0.1), which affects the randomness of the generated text.

---

### Prompts

A language model prompt is a user-provided set of instructions or input designed to guide the model's response. This aids the model in understanding the context and producing relevant output, whether it involves answering questions, completing sentences, or participating in a conversation


```python
customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""
style = """American English \
in a Times New Roman and respectful tone
"""

prompt = f"""Translate the text \
that is delimited by triple backticks \
into a style that is {style}.
text: ```{customer_email}```
"""
```

This section defines a customer email and a desired style. Then, it constructs a prompt string that instructs the language model to translate the text within triple backticks to the specified style.

---

### Parsers

```python
# Output Parsers
print(prompt)
response = get_completion(prompt)
print(response)
```

Here, the prompt string is printed, and the `get_completion` function is called with the prompt as input. The generated completion is then printed.

**Hugging Face Integration**

```python
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
```

This part integrates with the Hugging Face pipeline for text generation. It defines a function huggingface_completion that takes a prompt and optional parameters, and returns a completion generated by the specified model. Finally, it demonstrates the usage of this function by generating a completion for a specific prompt and printing it.

---

### Indexes

* Indexes in the context of language models (LLMs) refer to structured representations of documents that facilitate efficient interaction with the documents. These indexes play a crucial role in the retrieval of relevant documents in response to user queries.

* Imagine you have a vast collection of text documents. Without indexes, searching through this collection for relevant information would be like looking for a needle in a haystack. Indexes help organize this information in a way that makes it easier for LLMs to quickly find and retrieve the most relevant documents based on a user's query.

* The primary use of indexes in chains is in the retrieval step. This involves taking a user's query and using the index to identify and return the most relevant documents. However, indexes can be used for other purposes besides retrieval, and retrieval itself can employ other methods besides indexes to find relevant documents.

* It's important to note that indexes are typically used for unstructured data, such as text documents. For structured data like SQL tables or APIs, different methods are employed.

* LangChain primarily supports indexes and retrieval mechanisms centered around vector databases. These databases store documents as vectors, which enables efficient searching and retrieval based on similarities between vectors.

  - **Document Loading:** This is the first step where the raw data (documents) are loaded into the system. The documents could be in various formats such as text files, PDFs, HTML, etc.
 
  - **Document Splitting:** Once the documents are loaded, they are split into smaller chunks or segments. This is done to make the data more manageable and to improve the efficiency of the subsequent steps.

  - **Vectors and Embeddings:** Each chunk of data is then transformed into a vector representation, also known as an embedding. These embeddings capture the semantic meaning of the data and are used for efficient retrieval of relevant information. LangChain primarily supports indexes and retrieval mechanisms centered around vector databases.

  - **Retrieval:** This is the final step where a user’s query is taken and the system uses the index to identify and return the most relevant documents. The retrieval is based on the similarity between the query vector and the document vectors.

---

#### Document Loading

**Loading Environment Variable**

```python
from secret_key import hugging_facehub_key
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hugging_facehub_key
```

---

##### 1. PDF

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

**Summarizer Initialization**

```python
# Initialize summarizer
summarizer = HuggingFaceHub(
    repo_id="facebook/bart-large-cnn",
    model_kwargs={"temperature":0, "max_length":180}
)
```

* `summarizer:` An instance of `HuggingFaceHub` initialized with the BART-large model (`facebook/bart-large-cnn`) from the Hugging Face model hub.

* `model_kwargs:` Additional keyword arguments passed to the model during initialization, including `temperature` and `max_length`.

**Summarization Function**

```python
# Function to summarize text
def summarize(llm, text) -> str:
    return llm.invoke(f"Summarize this: {text}!")
```

`Summarize:` A function that takes a language model (`llm`) and text as input and returns a summarized version of the text using the model.

`llm.invoke:` Invokes the language model to generate a summary of the provided text.

**Page Summarization**

```python
# Summarize page 10
page = pages[10]
summary = summarize(summarizer, page.page_content)
print(summary)
print(summarize)
```

* `page:` Retrieves the content of the 10th page from the extracted pages.

* `summary:` Generates a summary of the content of the 10th page using the `summarize` function and the initialized `summarizer`.

* `print(summary):` Prints the generated summary.

* `print(summarize):` (Assuming this was intended to be `print(summarizer)`) Prints the `summarizer` object, which might have been unintentional.

---

##### 2. youtube

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
```

Here, we import necessary modules from LangChain for loading documents from various sources. Specifically, we import `GenericLoader` for loading documents, `OpenAIWhisperParser` for parsing text, and `YoutubeAudioLoader` for loading audio from YouTube.

```python
# Define the YouTube video URL
url = "https://www.youtube.com/watch?v=jGwO_UgTS7I"
# Define the directory to save the downloaded content
save_dir = "../docs/youtube/"
```

These lines define the YouTube video URL and the directory to save the downloaded content from the video.

```python
# Initialize the loader with YouTubeAudioLoader and OpenAIWhisperParser
loader = GenericLoader(
    YoutubeAudioLoader([url], save_dir),
    OpenAIWhisperParser()
)
```

Here, we initialize the document loader with `YoutubeAudioLoader` for loading audio content from the specified URL and `OpenAIWhisperParser` for parsing the audio content.

```python
docs = loader.load()
```

We load the documents using the initialized loader, which downloads the audio content from the YouTube video, transcribes it, and parses it into documents.

```python
content = docs[0].page_content[:500]
```

We extract the content from the first document and select the first 500 characters to display as an example.

```python
# Assuming 'summarizer' is an instantiated summarization model
# You can summarize the content using it
summary = summarize(summarizer, docs)
```

Here, assuming `summarizer` is an instantiated summarization model (like the one initialized previously), we use it to summarize the loaded documents.

```python
# Print the content and summary
print("Content:", content)
print("Summary:", summary)
```

Finally, we print the content and summary for demonstration purposes.

---

##### 3. URLs

This line imports the `WebBaseLoader` from LangChain, which is used to load documents from a web URL.

```python
from langchain.document_loaders import WebBaseLoader
```

Here, we initialize a `WebBaseLoader` object with the URL of a document hosted on the web. This loader will fetch the document from the specified URL.

```python
# Initialize the WebBaseLoader with the URL
loader = WebBaseLoader("https://github.com/tamaraiselva/git-demo/blob/main/metriales.docx")
```

This line loads the document(s) using the initialized loader.

```python
# Load documents
docs = loader.load()
```

We retrieve the content of the first document loaded. In this case, we only take the first 500 characters of the content for demonstration purposes.

```python
# Get the content of the first document
content = docs[0].page_content[:500]
```

Here, we summarize the content of the document(s) using a pre-instantiated summarization model named `summarizer`. However, there seems to be a slight issue here. The `summarize` function expects a single document's content, but we're passing the entire list of documents. It should likely be `summary = summarize(summarizer, content)` instead.

```python
# Assuming 'summarizer' is an instantiated summarization model
# You can summarize the content using it
summary = summarize(summarizer, docs)
```

Finally, we print the content of the document and its summary for inspection.

```python
# Print the content and summary
print("Content:", content)
print("Summary:", summary)
```

---

##### 4. NOTION

This line imports the `NotionDirectoryLoader` class from the `document_loaders` module in the LangChain framework. This loader is specifically designed to load documents from a directory containing Notion-exported Markdown files.

```python
from langchain.document_loaders import NotionDirectoryLoader
```

Here, we create an instance of the `NotionDirectoryLoader` class and provide the path to the directory where Notion-exported Markdown files are located. In this case, the directory is named "notion".

```python
loader = NotionDirectoryLoader("notion")
docs = loader.load()
```

We use the `load()` method of the `loader` instance to load the documents from the specified directory. This method returns a list of `Document` objects representing the loaded documents.

```python
if docs:
    print(docs[0].page_content[0:100])
    print(docs[0].metadata)
else:
    print("No Notion documents were loaded.")
```

This conditional statement checks if any documents were loaded. If there are documents, it prints the first 100 characters of the content of the first document (`docs[0].page_content[0:100]`) and the metadata of the first document (`docs[0].metadata`). If no documents were loaded, it prints a message indicating that no Notion documents were loaded.

---

#### Documnet Splitting

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
```

We define the parameters for text splitting.` chunk_size` specifies the maximum length of each chunk, and `chunk_overlap` specifies how much overlap there should be between adjacent chunks.

```python
chunk_size =26
chunk_overlap = 4
```

We create an instance of `CharacterTextSplitter` and initialize it with the specified parameters. Optionally, you can specify a `separator` if you want to split the text based on a particular character or string.

```python
# Initialize the CharacterTextSplitter
c_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separator=' '  # Optional, if you want to split by a separator
)
```

We define the text that we want to split into smaller chunks.

```python
# Define the text
text = 'abcdefghijklmnopqrstuvwxyzabcdefg'
```

We use the `split_text()` method of the `CharacterTextSplitter` instance to split the text into smaller chunks based on the specified parameters

```python
# Split the text using the CharacterTextSplitter
chunks = c_splitter.split_text(text)
```

Finally, we print the resulting chunks.

```python
print(chunks)
```

**Recursive splitting details**

**`Using RecursiveCharacterTextSplitter:`**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, NotionDirectoryLoader
```


```python
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

#### Vectors and Embeddings

**Embeddings**

```python
# !pip install sentence-transformers
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
```
This line is a comment indicating that you should install the sentence-transformers package if you haven't already. It's likely included as a reminder in case the package isn't installed in your environment.

Here, we import two different types of embedding models from LangChain: `OpenAIEmbeddings` and `HuggingFaceEmbeddings`. These models are used to generate embeddings for text.

```python
embeddings = HuggingFaceEmbeddings()
```

We initialize an embedding model using` HuggingFaceEmbeddings()`. This creates an instance of the Hugging Face embedding model.

```python
text = "This is a test document to check the embeddings."
text_embedding = embeddings.embed_query(text)
```
We define a sample text that we want to generate embeddings for.

We use the initialized embedding model (`embeddings`) to generate embeddings for the given text (`text`) using the `embed_query()` method.

```python
print(f'Embeddings lenght: {len(text_embedding)}')
print (f"Here's a sample: {text_embedding[:5]}...")
```

We print the length of the embeddings generated for the text and show a sample of the embeddings. The length indicates the dimensionality of the embeddings, and the sample provides a glimpse of the first few values of the embeddings.

**Vectorstore**

VectorStore is a component of LangChain that facilitates efficient storage and retrieval of document embeddings, which are vector representations of documents. These embeddings are created using language models and are valuable for various natural language processing tasks such as information retrieval and document similarity analysis.

**Installation:**

To install VectorStore, you can use pip:

```python
# ! pip install langchain-chroma
```
`Usage:`

```python
from langchain_chroma import Chroma
```

First, import the Chroma class from langchain_chroma module.

```python
db = Chroma.from_documents(splits, embeddings)
```
Then, create a VectorStore instance using the from_documents method. This method requires two parameters:

* ` splits: `A list of document splits, where each split represents a document.

* `embeddings:` A list of embeddings corresponding to the document splits.

```python
print(db._collection.count())
```
Finally, you can access the number of documents stored in the VectorStore using the count() method on the _collection attribute.

---

#### Retrevial


**Vectorstore retrieval**

**Installation:**

```python
# !pip install lark
# !pip install pypdf tiktoken faiss-cpu
```

This code block is commented out, but it suggests installing necessary packages using pip. However, since it's commented out, it doesn't affect the execution of the code. These packages seem to be dependencies for LangChain.

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
```

These lines import necessary modules from LangChain for document loading (`PyPDFLoader`), text splitting (`CharacterTextSplitter`, `RecursiveCharacterTextSplitter`), vector stores (`FAISS`), and embeddings (`HuggingFaceEmbeddings`).

```python
loader = PyPDFLoader("MachineLearning-Lecture01.pdf")
documents = loader.load()
```

Here, a `PyPDFLoader` instance is created to load a PDF document named "MachineLearning-Lecture01.pdf". The `load()` method is then used to extract the content of the document into a list of documents.

```python
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
```

A `CharacterTextSplitter` instance is created with specified parameters for chunk size and overlap. Then, the `split_documents()` method is used to split the documents into smaller text chunks based on the specified parameters.

```python
# Initialize the HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(texts, embeddings)
```

An instance of `HuggingFaceEmbeddings` is initialized. Then, a FAISS vector store (`FAISS`) is created from the text chunks using the embeddings obtained from the Hugging Face model.

```python
# You can also specify search kwargs like k to use when doing retrieval.
#retriever = db.as_retriever()
retriever = db.as_retriever(search_kwargs={"k": 2})
```

A retriever object is created from the FAISS vector store. Additional search arguments, such as the number of nearest neighbors (`k`), can be specified.

```python
print(len(documents))
```
This line prints the number of documents loaded from the PDF file.

---

### Memory

Memory which is still in beta phase is an essential component in a conversation. This allows us to infer information in past conversations. Users have various options, including preserving the complete history of all conversations, summarizing the ongoing conversation, or retaining the most recent n exchanges.

![image](img/memory.png)

```python
from langchain.memory import ChatMessageHistory
```

Here, we import the ChatMessageHistory class from LangChain's memory module. This class allows us to maintain a history of user and AI messages during a conversation.

```python
history = ChatMessageHistory()
```

We create an instance of `ChatMessageHistory` named `history` to store the chat message history.

```python
history.add_user_message("hi!")

history.add_ai_message("whats up?")
```

We add a user message "hi!" to the chat history using the `add_user_message` method.

Similarly, we add an AI message "whats up?" to the chat history using the `add_ai_message` method.

```python
history.messages

history.add_user_message("Fine, what about you?")
history.messages
```

We access the messages stored in the chat history. This will display both the user and AI messages added earlier.

Another user message "Fine, what about you?" is added to the chat history.
from langchain.chat_models import ChatHuggingFace

We access the updated chat history to view all the messages, including the latest user message.

```python
from langchain.chat_models import ChatHuggingFace
```

Here, we import the ChatHuggingFace class from LangChain's chat_models module. This class allows us to interact with Hugging Face models for conversational AI tasks.

```python
llm = ChatHuggingFace(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 512,
        "top_k": 30,
        "temperature": 0.1,
        "repetition_penalty": 1.03,
    },
)
```

We initialize an instance of the `ChatHuggingFace` class named `llm`. This instance is configured to use a specific model (`HuggingFaceH4/zephyr-7b-beta`) for text generation, with certain model parameters like `max_new_tokens`, `top_k`, `temperature`, and `repetition_penalty`.

```python
chat = ChatHuggingFace()
ai_response = chat(history.messages)
ai_response
```

We create another instance of `ChatHuggingFace` named `chat`. This instance will be used for generating AI responses based on the chat history.

We generate an AI response using the `chat` instance and pass the chat history (`history.messages`) as input. The AI model processes the history and generates a response.

```python
history.add_ai_message(ai_response.content)
history.messages
```

The AI response generated is added to the chat history using the `add_ai_message` method.

Finally, we access the updated chat history to view all the messages, including the latest AI response that was added.

----

### Chains

Chains form the backbone of LangChain's workflows, seamlessly integrating Language Model Models (LLMs) with other components to build applications through the execution of a series of functions.

The fundamental chain is the LLMChain, which straightforwardly invokes a model and a prompt template. For example, consider saving a prompt as "ExamplePrompt" and intending to run it with Flan-T5. By importing LLMChain from langchain.chains, you can define a chain_example like so: LLMChain(llm=flan-t5, prompt=ExamplePrompt). Executing the chain for a given input is as simple as calling chain_example.run("input").

For scenarios where the output of one function needs to serve as the input for the next, SimpleSequentialChain comes into play. Each function within this chain can employ diverse prompts, tools, parameters, or even different models, catering to specific requirements.

```python
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
```
Here, we import necessary modules from LangChain. We import `HuggingFaceHub` to utilize a pre-trained language model from the Hugging Face model hub, `PromptTemplate` to create a template for generating prompts, and `LLMChain` to create a chain for executing language model tasks.

```python
llm_hf = HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":64})
```

We initialize a language model from the Hugging Face model hub. In this case, we're using the T5 model (`google/flan-t5-xl`). We also provide some model-specific arguments like `temperature` and `max_length`.

```python
template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
```
We define a prompt template using the `PromptTemplate` class. The template contains a placeholder {`question`} for the input question. This template will be used to generate prompts for the language model.

```python
llm_chain = LLMChain(prompt=prompt, llm=llm_hf)
```

We create an `LLMChain` instance by providing the prompt template and the initialized language model (`llm_hf`). This chain will use the template to generate prompts for the language model.

```python
question = "Who won the FIFA World Cup in the year 1994? "
```

We define a question that we want to ask the language model.

```python
print(llm_chain.run(question))  
```

We run the defined question through the LLMChain by calling the run method and passing the question as input. This will generate a prompt using the template, execute it using the language model, and return the generated response.

---

### Agents

Agents, at their core, leverage a language model to make decisions about a sequence of actions to be taken. Unlike chains where a predefined sequence of actions is hard coded directly in the code, agents use a llm as a reasoning engine to determine the actions to be taken and their order.

```python
# !pip install google-search-results
from langchain import HuggingFaceHub
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
```

Here, we import necessary modules from LangChain. We also install a Python package called google-search-results, which seems to be required but is commented out.

```python
llm_hf = HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":64})
llm =  PromptLayer(temperature=0)
search = HuggingFaceHub()
```

These lines initialize different language models using HuggingFaceHub and PromptLayer. The `llm_hf` model is initialized with the Google Flan T5 XL model, while the `llm` model is initialized with the PromptLayer model.

```python
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search"
    )
]
```

A list of tools is defined, with each tool containing a name, function, and description. In this case, the tool is named "Intermediate Answer" and uses the `search.run` function.

```python
self_ask_with_search = initialize_agent(tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True)
```

An agent named `self_ask_with_search` is initialized using the `initialize_agent` function. The agent uses the specified tools, language model (`llm`), and agent type (`AgentType.SELF_ASK_WITH_SEARCH`).

```python
self_ask_with_search.run("What is the hometown of the reigning men's French Open?")
```

We run the agent with a specific question as input. The agent will likely use its internal logic to provide an answer based on the question and the available tools.

```python
#if you want the intermediate answers, pass return_intermediate_steps=True.
self_ask_with_search = initialize_agent(tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, return_intermediate_steps=True, verbose=True)
response = self_ask_with_search("What is the hometown of the reigning men's French Open?")

import json
print(json.dumps(response["intermediate_steps"], indent=2))
```

Here, we reinitialize the agent with an additional parameter `return_intermediate_steps=True`, which indicates that we want to capture intermediate steps during the agent's processing. Then, we run the agent again with the same question and print out the intermediate steps in a JSON format.

---

## References

if to learn click this link button 👇

1. [Reference link](https://github.com/sudarshan-koirala/youtube-stuffs/blob/main/langchain/LangChain_Components.ipynb)

2. [Reference link](https://github.com/PradipNichite/Youtube-Tutorials/blob/main/Youtube_Course_Sentence_Transformers.ipynb)

3. [Reference link](https://www.youtube.com/watch?v=jbFHpJhkga8&list=PLz-qytj7eIWVd1a5SsQ1dzOjVDHdgC1Ck)

4. [Reference link](https://www.youtube.com/watch?v=nAmC7SoVLd8list=PLeo1K3hjS3uu0N_0W6giDXzZIcB07Ng_F)

5. [Reference link](https://www.youtube.com/watch?v=mBJqGAHoam4)

6. [Reference link](https://langchain-cn.readthedocs.io/en/latest/modules/models/text_embedding/examples/huggingfacehub.html)