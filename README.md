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

## Introduction to Langchain

* LangChain is an innovative open-source framework designed to empower developers in building cutting-edge applications utilizing LLMs, or Large Language Models. These models, like the one I'm based on, possess advanced capabilities in natural language understanding and generation, making them ideal for a wide range of tasks such as text generation, summarization, translation, and more. 

* LangChain provides a structured environment and tools that streamline the integration of LLMs into various applications, facilitating rapid development and deployment. With LangChain, developers can leverage the power of LLMs to create intelligent and dynamic solutions tailored to their specific needs, driving innovation in natural language processing technology.

### Evolve 

**`Framework Enhancements:`** LangChain itself can evolve by receiving updates, improvements, and new features. This evolution might involve optimizing performance, adding support for new LLM architectures, enhancing compatibility with various programming languages and platforms, and incorporating feedback from the developer community.

**`Model Training:`** Developers can use LangChain to evolve LLMs by training them on new datasets or fine-tuning existing models for specific tasks or domains. This process involves feeding the model additional data to improve its accuracy, adaptability, and language understanding capabilities.

**`Application Development:`** Developers can evolve their applications built with LangChain by continuously refining features, adding new functionalities, and improving user experiences. This iterative process allows applications to adapt to changing user needs, technological advancements, and market trends.

**`Community Collaboration:`** The LangChain community can evolve through collaboration, knowledge sharing, and the exchange of ideas. Developers can contribute to the framework's development, share their experiences and insights, and collectively address challenges and opportunities in LLM-based application development.


### Why do we need Langchain

* LangChain involves preprocessing the text corpus by dividing it into manageable chunks or summaries, embedding them in a vector space, and subsequently searching for similar chunks when a question is posed

* LangChain simplifies the composition of these components by offering an abstraction.

    **`Efficient Integration:`** Integrating LLMs into applications can be complex  and time-consuming. LangChain streamlines this process by providing a framework  specifically tailored for working with LLMs, reducing development time and effort.


    **`Scalability:`** As LLMs become increasingly large and sophisticated, managing    their integration and scalability becomes more challenging. LangChain offers   solutions to effectively handle the scalability of LLM-powered applications,  ensuring they can handle growing volumes of data and users.


    **`Customization:`** Different applications have unique requirements and use    cases. LangChain enables developers to customize and fine-tune LLMs to suit their  specific needs, whether it's optimizing for performance, adapting to     domain-specific vocabularies, or incorporating specialized features.


    **`Open-Source Community:`** By being open-source, LangChain encourages     collaboration and innovation within the developer community. Contributors can   enhance the framework, fix bugs, add features, and share best practices,  fostering a vibrant ecosystem around LLM development.


    **`Standardization:`** LangChain promotes standardization in LLM development    practices, making it easier for developers to understand and work with different   models. This standardization enhances interoperability between different  LLM-based applications and facilitates knowledge sharing among developers.

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

### Schema



### Models

`Loading Environment Variable`

```python
import promptlayer
import os
os.environ["PROMPTLAYER_API_KEY"] = "pl_460558b60a94f36bb723dfdbd5409642"
```

Here, the code imports the `promptlayer` module and sets the environment variable `PROMPTLAYER_API_KEY` to a specific API key. This API key is likely used for accessing a service that provides language model capabilities.

`Setting Model Variable`

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


`Defining Completion Function`

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

### Prompts

`Defining Prompt`

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


### Parsers

```python
# Output Parsers
print(prompt)
response = get_completion(prompt)
print(response)
```

Here, the prompt string is printed, and the `get_completion` function is called with the prompt as input. The generated completion is then printed.


`Hugging Face Integration`

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

### Indexes

* Indexes in the context of language models (LLMs) refer to structured representations of documents that facilitate efficient interaction with the documents. These indexes play a crucial role in the retrieval of relevant documents in response to user queries.

* Imagine you have a vast collection of text documents. Without indexes, searching through this collection for relevant information would be like looking for a needle in a haystack. Indexes help organize this information in a way that makes it easier for LLMs to quickly find and retrieve the most relevant documents based on a user's query.

* The primary use of indexes in chains is in the retrieval step. This involves taking a user's query and using the index to identify and return the most relevant documents. However, indexes can be used for other purposes besides retrieval, and retrieval itself can employ other methods besides indexes to find relevant documents.

* It's important to note that indexes are typically used for unstructured data, such as text documents. For structured data like SQL tables or APIs, different methods are employed.

* LangChain primarily supports indexes and retrieval mechanisms centered around vector databases. These databases store documents as vectors, which enables efficient searching and retrieval based on similarities between vectors.

#### Document Loading

**Loading Environment Variable**

```python
from secret_key import hugging_facehub_key
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hugging_facehub_key
```

##### PDF

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


**output**

```output
Many students will try to build a cool machine learning application. Some students try to improve state-of-the-art machine learning algorithms. Some of those projects are also very successful. And a smaller minority of students sometimes try to prove theorems about machine learning.
```

##### youtube

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


##### URLs

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

##### NOTION

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

`Purpose:` This loads documents from a directory, assuming they are stored in Notion's format.

`Parameters:`

- `"directory_path":` The path to the directory containing Notion documents.

`Returns:` A list of document objects representing the contents of each document in the directory.


**`Splitting Documents`**

```python
docs = c_splitter.split_documents(pages)
```

`Purpose:` This splits loaded documents into chunks using the defined text splitter.

`Parameters:`

- `pages:` The list of document objects to be split.

`Returns:` A list of document chunks, each representing a portion of the original document.

#### Vectors and Embeddings

**`Embeddings`**

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


**`Vectorstore`**

VectorStore is a component of LangChain that facilitates efficient storage and retrieval of document embeddings, which are vector representations of documents. These embeddings are created using language models and are valuable for various natural language processing tasks such as information retrieval and document similarity analysis.

`Installation:`

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

-` splits: `A list of document splits, where each split represents a document.

- `embeddings:` A list of embeddings corresponding to the document splits.

```python
print(db._collection.count())
```
Finally, you can access the number of documents stored in the VectorStore using the count() method on the _collection attribute.

#### Retrevial


**`Vectorstore retrieval`**

`Installation:`

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

### Memory



### Chains



### Agents



## References

if to learn click this link button 👇

1. [Reference link](https://github.com/sudarshan-koirala/youtube-stuffs/blob/main/langchain/LangChain_Components.ipynb)

2. [Reference link](https://github.com/PradipNichite/Youtube-Tutorials/blob/main/Youtube_Course_Sentence_Transformers.ipynb)

3. [Reference link](https://www.youtube.com/watch?v=jbFHpJhkga8&list=PLz-qytj7eIWVd1a5SsQ1dzOjVDHdgC1Ck)

4. [Reference link](https://www.youtube.com/watch?v=nAmC7SoVLd8list=PLeo1K3hjS3uu0N_0W6giDXzZIcB07Ng_F)

5. [Reference link](https://www.youtube.com/watch?v=mBJqGAHoam4)

6. [Reference link](https://langchain-cn.readthedocs.io/en/latest/modules/models/text_embedding/examples/huggingfacehub.html)