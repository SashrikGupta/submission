from astrapy import DataAPIClient
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
from langchain_community.tools import TavilySearchResults
import os 
from langchain.agents import initialize_agent, Tool
import requests
import google.generativeai as genai
import os
import google.generativeai as genai
import json 
import typing_extensions as typing
import requests
import uuid 
import concurrent.futures
import base64 
import time 

BASE_API_URL = "https://api.langflow.astra.datastax.com"
LANGFLOW_ID = "e20db916-b085-4204-bf27-6554b022d84c"
FLOW_ID = "aab6cb31-a9e6-4849-a368-341e8344eb90"
APPLICATION_TOKEN = "AstraCS:TANlpNPNQzBYxeZqNlxTCbre:e19e30c453a0454ea37c912ad6a0adbc6f23c6ec4156538fb8791dc9857e2e88"

os.environ['GOOGLE_API_KEY'] = "AIzaSyAPB3pWq1AbDtw7W5g3p1AGDzKivvRmlHY"
genai.configure(api_key="AIzaSyAPB3pWq1AbDtw7W5g3p1AGDzKivvRmlHY")
# Initialize the Astra client
client = DataAPIClient("AstraCS:zgJSwvWzsZXjawoPFyNcXMNS:9eed3b61fbbf2bf615a766b0a661bf597de4340be47436100e3e84fd1b38fa14")

db = client.get_database_by_api_endpoint(
    "https://fb32a9e0-4a7c-48a5-9b2b-0a993c2abdf4-westus3.apps.astra.datastax.com"
)
social_collection = db["some_data"]

documents = social_collection.find({})
data = list(documents)
df = pd.DataFrame(data)

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="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"
             
)


llm_agent = create_pandas_dataframe_agent(llm, df,verbose=True , allow_dangerous_code=True , handle_parsing_errors=True)

os.environ["TAVILY_API_KEY"] = "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"
# Initialize TavilySearchResults
tavily_search = TavilySearchResults(max_results=2)

# Define the tool
tools = [
    Tool(
        name="TavilySearch",
        func=tavily_search.run,
        description="Search for information using Tavily search results."
    )
]


# Create the agent with TavilySearchResults
internet_agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,
)

def rag_bot(prompt): 
   client = DataAPIClient("AstraCS:KWMZZzdblHwQbvfjBjgoXDJQ:c6bbe014f5282953ab9408ce0c8debcf10e872a4816447c47dac9576072a41f0")
   db = client.get_database_by_api_endpoint(
   "https://d6d58343-23e4-4b22-a6ee-651c46d85a72-us-east-2.apps.astra.datastax.com"
   )

   collection = db["goog_rag"]
   query = prompt
   results = collection.find(
      sort={"$vectorize": query},
      limit=2,
      projection={"$vectorize": True},
      include_similarity=True,
   )

   context = ''
   for document in results:
      context+=document['$vectorize']



   return llm.invoke(f""" you are a social media influencer's manager given the uerse prompt
                     
            {prompt}
      give answer on the basis of below context : 
            {context}
   """).content

def db_bot(message):
    api_url = f"{BASE_API_URL}/lf/{LANGFLOW_ID}/api/v1/run/{FLOW_ID}"

    payload = {
        "input_value": message,
        "output_type": "chat",
        "input_type": "chat",
        "session_id" : uuid.uuid4().hex
    }
    headers = None

    headers = {"Authorization": "Bearer " + APPLICATION_TOKEN, "Content-Type": "application/json"}
    response = requests.post(api_url, json=payload, headers=headers)
    print(response)
    response = response.json()
    response = response['outputs'][0]['outputs'][0]['results']['message']['text']

    return response


def photo_bot(image_path):
    genai.configure(api_key="AIzaSyC9KkbgmUDIB8BbiaKDmjrxTVI1omRh-TQ")
    with open(image_path, "rb") as image_file:
        image_content = image_file.read()
    image_base64 = base64.b64encode(image_content).decode('utf-8')
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    prompt = """
   you are an social media infulencer's manager do the following : 
   comprehensively summarize the image ,
   give pros and cons of the image advising the influencer and also 
   give some suggestions 
"""
    response = model.generate_content([{'mime_type': 'image/jpeg', 'data': image_base64}, prompt])
    return response.text

def video_bot(video_file_path):
   genai.configure(api_key="AIzaSyC9KkbgmUDIB8BbiaKDmjrxTVI1omRh-TQ")
   print(f"Uploading file...")
   time.sleep(10)
   video_file = genai.upload_file(path=video_file_path)
   print(f"Completed upload: {video_file.uri}")
   prompt = """
   you are an social media infulencer do the following : 
   comprehensively summarize the video ,
   give pros and cons of the vieo advising the influencer and also 
   give some suggestions 
   """
   model = genai.GenerativeModel(model_name="gemini-1.5-pro")
   print("Making LLM inference request...")
   response = model.generate_content([video_file, prompt],
                                  request_options={"timeout": 600})
   return response.text


def run_flow(prompt) : 
   class Prompt(typing.TypedDict):
    agent_name: str
    prompt: str



   model = genai.GenerativeModel("gemini-1.5-pro-latest")
   result = model.generate_content(
      f"""

   Your Goal: For each type of query (database, internet, book ,graph_visualization), ensure the prompts are clear, concise, and optimized for extracting accurate and valuable insights. Use a professional tone and include context where necessary.

   Given a user's query : 

   {prompt} 

   and the database table structure, create a refined and detailed prompt that another LLM or agent can use to retrieve meaningful insights from the database.
   The database contains influencer-related data with attributes such as Post_ID, Post_Type, Post_Content, Post_Timestamp, Likes, Comments, Shares, Impressions, Reach, Engagement_Rate, Audience_Age, Audience_Gender, Audience_Location, Audience_Interests, and Sentiment.
   Example: If the user asks, "What is the engagement rate of posts with 'Video' content from Instagram targeting Paraguay's audience?" your task is to produce a well-structured prompt referencing the table attributes to extract such data.
   Internet Query Prompt Creation

   Transform the user's query into a clear, focused prompt that another LLM or agent can use to search the internet effectively for relevant insights.
   Example: If the user asks, "How do influencers increase engagement on Instagram?" craft a refined prompt that guides the agent to gather the most relevant and comprehensive information.
   Book-Related Content Prompt Creation

   Craft a prompt to extract meaningful studies, topics, or content related to the user's query from The Art of Social Media by Guy Kawasaki and Peg Fitzpatrick.
   Example: If the user asks, "How to optimize social media posts for better audience reach?" your task is to create a focused prompt for finding relevant sections, insights, or strategies in the book.
   Format for Prompt Output
   For every query, output the crafted prompts in the following format:


   give a very very very short prompt for generation of graphs related to the usres query and note : just ask one prompt here 

   examples : 
   if query is related to  "Analysis of the relation between sentiment and gender."
   Prompt: "heatmap between gender and sentiment."
   if query is related to  "Analysis of the age distribution."
   Prompt: "histogram for age distribution."
   if query is related to " distribution of images, videos, and links?"
   Prompt: "pie chart of POST_TYPE."
   if query is related to "How does reach vary across audience interests?"
   Prompt: "bar graph of reach by audience interests."


   User:  
   [Provide the user's original query here.]  

   Database:  
   [Provide the refined prompt tailored to querying the database.]  

   Internet:  
   [Provide the refined prompt tailored to querying the internet for insights.]  

   RAG:  
   [Provide the refined prompt tailored to retrieving relevant content from *The Art of Social Media*.]  

   Graph : 
   [give a very short prompt for generation of graphs]


   NOTE THERE ARE 5 AGENTS : 

   User , Database , Internet , RAG , Graph

   Examples

   Example 1

   [
      {{ 
         agent_name : User  ,
         prompt  : "What is the engagement rate of Instagram video posts targeting Paraguay's audience?"
         
      }}
      , 
      {{
         agent_name : Database , 
         prompt : "Retrieve insights from the database table about engagement rates (Engagement_Rate) for 'Video' posts (Post_Type) on Instagram (Post_Platform) targeted at audiences in Paraguay (Audience_Location). Include aggregated statistics if possible."
      }}
      , 
      {{
         agent_name : Internet , 
         prompt : "Search for statistics or strategies related to the engagement rate of Instagram video posts targeted at audiences in Paraguay. Focus on influencer content performance."
      }}
      ,
      {{
         agent_name : RAG , 
         prompt : "From The Art of Social Media by Guy Kawasaki and Peg Fitzpatrick, find insights or strategies related to improving the engagement rate of video posts on Instagram for specific audience demographics, such as those in Paraguay."
      }}
      ,
      {{
         agent_name : Graph , 
         prompt : "bar graph of egagement rate vs audience_location"
      }}
   ]


   Example 2

   [
      {{
         agent_name : User , 
         prompt : "How do influencers increase engagement on Instagram?"
      }}
      ,
      {{
         agent_name : Database , 
         prompt : "Identify trends or factors from the database table that contribute to increased engagement (Likes, Comments, Shares, Engagement_Rate) for Instagram (Post_Type) posts."
      }}
      ,
      {{
         agent_name : Internet , 
         prompt : "Find strategies used by influencers to increase engagement on Instagram, focusing on likes, comments, shares, and impressions. Include tools and best practices."
      }}
      ,
      {{
         agent_name : RAG ,
         "From The Art of Social Media, extract strategies or case studies on increasing engagement on Instagram. Highlight actionable tips, tools, or examples."
      }}
      ,
      {{
         agent_name : Graph , 
         prompt : "null"
      }}

   ]




      """,
      generation_config=genai.GenerationConfig(
         response_mime_type="application/json", response_schema=list[Prompt]
      ),
   )
   result = json.loads(result.candidates[0].content.parts[0].text)

   print(result)


   answer_database = "no information, some error occurred"
   answer_internet = "no information, some error occurred"
   answer_rag = "no information, some error occurred"
   graph_path = None

   def fetch_database():
        try:
            return db_bot(result[1]['prompt'])
        except Exception as e:
            return f"Error fetching database: {e}"

   def fetch_internet():
        try:
            return internet_agent.run(result[2]['prompt'] + " note at max do only one search, call the tool only one time and whatever you find return it as the final answer")
        except Exception as e:
            return f"Error fetching internet data: {e}"

   def fetch_rag():
        try:
            return rag_bot(result[3]['prompt'])
        except Exception as e:
            return f"Error fetching RAG data: {e}"

   def generate_graph():
        try:
            llm_agent.invoke(f"generate a {result[4]['prompt']} by seaborn graph and store in ./data as 'graph.jpg' in static image format")
            return './data/graph.jpg'
        except Exception as e:
            return f"Error generating graph: {e}"

   with concurrent.futures.ThreadPoolExecutor() as executor:
        future_database = executor.submit(fetch_database)
        future_internet = executor.submit(fetch_internet)
        future_rag = executor.submit(fetch_rag)
        future_graph = executor.submit(generate_graph)

        # Collect results
        answer_database = future_database.result()
        answer_internet = future_internet.result()
        answer_rag = future_rag.result()
        graph_path = future_graph.result()

   answer = {
        'database': answer_database,
        'internet': answer_internet,
        'rag': answer_rag,
        'graph': "./data/graph.jpg",
        'prompts': result
    }

   final_response = model.generate_content(f"""

   following are some insight : 
                                    
   database_insight :
      {answer['database']}  

   internet_insight :
      {answer['internet']}

   rag_insight :
      {answer['rag']}


   you task is to beautiy the content to make this atteractive and add emojis while leaving no points 
   each and every point should be covered make the important points bold like for eg mathematical figures like 20% , 3rd etc...

   also you have to give your overview on the entire data and also a title 

   this will be the structure : 

   [title] 

   **section 1 : overview** 
   your overview on the data provided to you (short and simple of one para only inlcude all the mathematicall insights you got if any )

   **section 2 : what your story tells us** 
   every content of database insight should be wrriten just write in a attractive manner 

   **section 3 : what the internet tells us** 
   every content of internet insight should be wrriten just write in a attractive manner

   **section 4 : what theory tells us** 
   every content of rag_insight should be wrriten just write in a attractive manner



   give an attractive and engaging  marked down format of the above 
   """)

   return{
     'database' : answer_database , 
     'internet' : answer_internet , 
     'rag' : answer_rag ,
     'graph' : './data/graph.jpg'  , 
     'prompts' : result , 
     'output' : final_response.text
   }

def photo_flow(image_path) : 
   model = genai.GenerativeModel("gemini-1.5-pro-latest")
   answer = photo_bot(image_path)

   database_insight = db_bot(f"""                        
   the following is the summary of a post of your influencer 
                             {answer} 
   from theis post find some database insight that is related to this post 
   give large number of mathematicall stattisics from database that comments on the 
   video can be possitive can be negative , suggest hiw this type of content will be
   either benifical or problemati based on inisghts from databse 
   """)

   internet_insight_1 = internet_agent.invoke(f"""                        
   the following is the summary of a post of your influencer 
                             {answer} 
   from theis post find some internet insight that is related to this post 
   give large number of mathematicall stattisics from internet that comments on the 
   video can be possitive can be negative , suggest hiw this type of content will be
   either benifical or problemati based on inisghts from intenet  

   !!" note at max do only one search , call the tool only one time and whatever you find return it as the final answer"

   if you don't find any thing that helps just write whatever you found
   """)

   internet_insight_2 = internet_agent.invoke(f"""                        
   the following is the summary of a post of your influencer 
                             {answer} 
   find the current topics related to this gerne only and suggest top trending topics and 
   which lies in same domain and user can create video on it  

   !!" note at max do only one search , call the tool only one time and whatever you find return it as the final answer"

   if you don't find any thing that helps just write whatever you found
   """)
   final_response = model.generate_content(f"""

   following are some insight : 
                                    
   database_insight :
      {database_insight}  

   statistical_insight:
      {internet_insight_1}

   trending_topic_insight :
      {internet_insight_2}


   you task is to beautiy the content to make this atteractive and add emojis while leaving no points 
   each and every point should be covered make the important points bold like for eg mathematical figures like 20% , 3rd etc...

   also you have to give your overview on the entire data and also a title 

   this will be the structure : 

   [title] 

   **section 1 : overview** 
   your overview on the data provided to you (short and simple of one para only inlcude all the mathematicall insights you got if any )

   **section 2 : what your story tells us** 
   every content of database insight should be wrriten just write in a attractive manner 

   **section 3 : what the internet tells us** 
   every content of internet insight should be wrriten just write in a attractive manner

   **section 4 : whats happening around the globe** 
   every content of trending_topic_insigt should be wrriten just write in a attractive manner



   give an attractive and engaging  marked down format of the above 
   """)
   return final_response 
   

def video_flow(video_path) : 
   model = genai.GenerativeModel("gemini-1.5-pro-latest")
   answer = video_bot(video_path)

   database_insight = db_bot(f"""                        
   the following is the summary of a post of your influencer 
                             {answer} 
   from theis post find some database insight that is related to this post 
   give large number of mathematicall stattisics from database that comments on the 
   video can be possitive can be negative , suggest hiw this type of content will be
   either benifical or problemati based on inisghts from databse 
   """)

   internet_insight_1 = internet_agent.invoke(f"""                        
   the following is the summary of a post of your influencer 
                             {answer} 
   from theis post find some internet insight that is related to this post 
    suggest how this type of content will be
   either benifical or problemati based on inisghts from intenet  

   !!" note at max do only one search , call the tool only one time and whatever you find return it as the final answer"
   """)

   internet_insight_2 = internet_agent.invoke(f"""                        
   the following is the summary of a post of your influencer 
                             {answer} 
   find the current topics related to this gerne only and suggest top trending topics and 
   which lies in same domain and user can create video on it  

   !!" note at max do only one search , call the tool only one time and whatever you find return it as the final answer"
   """)
   final_response = model.generate_content(f"""

   following are some insight : 
                                    
   database_insight :
      {database_insight}  

   statistical_insight:
      {internet_insight_1}

   trending_topic_insight :
      {internet_insight_2}


   you task is to beautiy the content to make this atteractive and add emojis while leaving no points 
   each and every point should be covered make the important points bold like for eg mathematical figures like 20% , 3rd etc...

   also you have to give your overview on the entire data and also a title 

   this will be the structure : 

   [title] 

   **section 1 : Summary** 
   {answer} 
   --> make the summary short but should hold all the points 

   **section 2 : overview** 
   your overview on the data provided to you (short and simple of one para only inlcude all the mathematicall insights you got if any )

   **section 3 : what your story tells us** 
   every content of database insight should be wrriten just write in a attractive manner 

   **section 4 : what the internet tells us** 
   every content of internet insight should be wrriten just write in a attractive manner

   **section 5 : whats happening around the globe** 
   every content of trending_topic_insigt should be wrriten just write in a attractive manner


   give an attractive and engaging  marked down format of the above 
   """)
   return final_response 