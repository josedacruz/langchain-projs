import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


# GitHub Models (FREE TIER)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.environ.get("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com",
    temperature=0.8
)

messages = [
    SystemMessage(content="You are a helpful assistant that writes professional emails."),
    HumanMessage(content="""
    Write a professional email to a customer explaining that their order 
    #12345 has been delayed by 2 days due to weather conditions, 
    but will arrive by Friday. Include an apology and a 10% discount code 
    for their next purchase.
    """)
]

response = llm.invoke(messages)

print("Generated Email:")
print("-" * 50)
print(response.content)