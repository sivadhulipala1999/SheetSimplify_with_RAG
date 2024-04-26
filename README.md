# SheetSimplify with RAG LLMs 
The aim of this project is to simplify data retrieval from Excel Sheets using RAG LLMs, hence the name! Many organizations currently store their data in Excel sheets and have stored decades' worth of data in them. However, retrieving data from these sheets becomes quite difficult unless the user has some technical background. The idea of Natural Language Querying (NLQ) is to exactly solve this issue by allowing users to ask simple questions to a model and get appropriate and rational responses. This NLQ can be achieved using RAG LLMs, which is what we aim to build in this project. 

# Milestones to achieve
Since the project is ongoing, I would like to include what I am currently working on with LLMs. 
- Change the Prompt: The rag_llm.ipynb uses the existing gpt-3.5 turbo model directly without changing the prompts. I would like to update the prompt to include a more accurate model for this case
- Explore further features: Could we use LLMs to create out-of-the-box visualizations? Are there other features that we could bring forth in this project?  
- Production Ready: For this project, I would like to containerize the application. However, I would also like to utilize MLFlow for this case. Exploration in this case is ongoing. 

## References 
1. <a href="https://blog.langchain.dev/summarizing-and-querying-data-from-excel-spreadsheets-using-eparse-and-a-large-language-model/">LangChain's Blogpost on Retrieval from Excel Sheet</a>
2. <a href="https://www.youtube.com/watch?v=xQ3mZhw69bc&ab_channel=SamWitteveen">This YouTube Video explaining how to use it</a>
