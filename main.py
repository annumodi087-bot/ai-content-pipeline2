from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import time

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
    temperature=0.7
)

parser = StrOutputParser()

print("=" * 50)
print("   AI CONTENT PIPELINE - Day 6 Upgraded")
print("=" * 50)

topic = input("\nEnter a topic: ")
print("\nRunning pipeline...\n")

# Track total time
total_start = time.time()

# --- STEP 1: RESEARCH ---
start = time.time()
research_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert researcher. Be concise and factual."),
    ("human", "Give me 5 key points about {topic}. Just bullet points, no intro.")
])
research_chain = research_prompt | llm | parser
key_points = research_chain.invoke({"topic": topic})
print(f"STEP 1 - Key Points: ({round(time.time()-start, 1)}s)")
print(key_points)
print()

# --- STEP 2: BLOG POST ---
start = time.time()
blog_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional blog writer. Write engaging content."),
    ("human", "Write a short blog post (3 paragraphs) about {topic} using these key points:\n{key_points}")
])
blog_chain = blog_prompt | llm | parser
blog_post = blog_chain.invoke({"topic": topic, "key_points": key_points})
print(f"STEP 2 - Blog Post: ({round(time.time()-start, 1)}s)")
print(blog_post)
print()

# --- STEP 3: SUMMARY ---
start = time.time()
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at summarizing content clearly."),
    ("human", "Summarize this blog post in exactly 3 sentences:\n{blog_post}")
])
summary_chain = summary_prompt | llm | parser
summary = summary_chain.invoke({"blog_post": blog_post})
print(f"STEP 3 - Summary: ({round(time.time()-start, 1)}s)")
print(summary)
print()

# --- STEP 4: SOCIAL CAPTIONS ---
start = time.time()
social_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a social media expert who writes viral captions."),
    ("human", """Create 3 social media captions for this summary:
{summary}

Format:
LinkedIn: [professional caption]
Twitter: [short punchy caption under 280 chars]
Instagram: [caption with emojis]""")
])
social_chain = social_prompt | llm | parser
captions = social_chain.invoke({"summary": summary})
print(f"STEP 4 - Social Captions: ({round(time.time()-start, 1)}s)")
print(captions)
print()

# --- SAVE TO FILE ---
total_time = round(time.time() - total_start, 1)

filename = f"content_{topic[:20].replace(' ','_')}.txt"
with open(filename, "w", encoding="utf-8") as f:
    f.write(f"TOPIC: {topic}\n")
    f.write("=" * 50 + "\n\n")
    f.write("KEY POINTS:\n")
    f.write(key_points + "\n\n")
    f.write("BLOG POST:\n")
    f.write(blog_post + "\n\n")
    f.write("SUMMARY:\n")
    f.write(summary + "\n\n")
    f.write("SOCIAL CAPTIONS:\n")
    f.write(captions + "\n")

print("=" * 50)
print(f"Pipeline complete in {total_time}s!")
print(f"Saved to: {filename}")
print("=" * 50)