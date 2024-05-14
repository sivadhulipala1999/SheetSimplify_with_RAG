# SheetSimplify with RAG LLMs 
The aim of this project is to simplify data retrieval from Excel Sheets using RAG LLMs, hence the name! Many organizations currently store their data in Excel sheets and have stored decades' worth of data in them. However, retrieving data from these sheets becomes quite difficult unless the user has some technical background. The idea of Natural Language Querying (NLQ) is to exactly solve this issue by allowing users to ask simple questions to a model and get appropriate and rational responses. This NLQ can be achieved using RAG LLMs, which is what we aim to build in this project. 

# The approach 
Instead of fine-tuning the model on the relevant data, which consumes significant resources, we shall attempt to utilize prompt-engineering to make the LLM answer based on the context provided via the dataset. This is the basic idea behind RAG. 

Once the model is done, we shall expose a simple API endpoint which responds with a summary of key information from the data. The user would also like to query the model further, for which we provide a streamlit app. 

Since we would like the whole application to be distributed, we would 'dockerize' it. 

# The repo 
The repo structure is based on a standard template for production ML projects [3]. 

<b>Notebooks</b>: Typically used for data analysis and exploration. Since we are dealing with LLMs, I decided to add my understanding of different concepts to the notebooks here. <br/>
<b>Model</b>: Contains the python file that preps and invokes the LLM.<br/> 
<b>API</b>: Contains the flask file which exposes the endpoint to make a simple call to the LLM. Following are the endpoints exposed so far <br/> 
    &emsp;&emsp;- "/" - home page which just says "Welcome to Sheet Simplify!" <br/>
    &emsp;&emsp;- "/v1/summary" - which provides a summary of the data provided as per the LLM  <br/>
<b>streamlit</b>: Contains the code to setup the streamlit app. Not in  the original template. 

There are several other folders from the original template which were not relevant to this case and hence have been omitted. 

# Usage Tips  
1. Before running the scripts in this repo, it is very important to perform a pip install on the entire project so that the internal packages become available to each other. To do this, run the following,
               <code>pip install .</code>
2. Regardless of the script you want to run, it is very important to execute it from the root directory. For example, you would run the llm.py file as follows from the root directory 
               <code>python -m model/llm.py</code>
3. For the streamlit app, run the following command
               <code>streamlit run streamlit/streamlit_app.py</code>
   This initiates the streamlit web app frontend in the localhost
4. You can also execute the flask app to hit the API endpoints using the command
                <code>python =m api/app.py</code>
   This initiates the flask server in localhost. Remember, you would have to add the endpoint /v1/summary to the URL in the browser to hit the endpoint.
   Of course, you can also hit these endpoints from other API tools like Postman. 

# Milestones to achieve
Since the project is ongoing, I would like to include what I am currently working on with LLMs. 
- Production Ready: I would like to utilize MLFlow for this case, if sensible. Exploration in this case is ongoing. 

## References 
1. <a href="https://blog.langchain.dev/summarizing-and-querying-data-from-excel-spreadsheets-using-eparse-and-a-large-language-model/">LangChain's Blogpost on Retrieval from Excel Sheet</a>
2. <a href="https://www.youtube.com/watch?v=xQ3mZhw69bc&ab_channel=SamWitteveen">This YouTube Video explaining how to use it</a>
3. <a href="https://github.com/DanielhCarranza/ml-production-template">Production ML Project Template</a>
4. <a href="https://huggingface.co/blog/open-source-llms-as-agents">Open Source LLMs as LangChain Agents</a> 
5. <a href="https://huggingface.co/learn/cookbook/advanced_rag">Advanced RAG tutorial from HuggingFace</a> 
