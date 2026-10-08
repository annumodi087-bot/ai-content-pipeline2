import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="AI Content Pipeline", page_icon="✍️", layout="wide")
st.title("✍️ AI Content Pipeline")
st.caption("Research → Blog → Summary → Social Captions | By Annu")

api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")

@st.cache_resource
def get_pipeline():
    llm = ChatGroq(api_key=api_key, model="openai/gpt-oss-120b", temperature=0.7)
    parser = StrOutputParser()

    research_prompt = ChatPromptTemplate.from_template(
        "Give 5 key facts about: {topic}"
    )
    blog_prompt = ChatPromptTemplate.from_template(
        "Write a 200 word blog post using these facts:\n{research}"
    )
    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize in 2 sentences:\n{blog}"
    )
    captions_prompt = ChatPromptTemplate.from_template(
        "Write 3 social media captions for:\n{summary}"
    )

    research_chain = research_prompt | llm | parser
    blog_chain = blog_prompt | llm | parser
    summary_chain = summary_prompt | llm | parser
    captions_chain = captions_prompt | llm | parser

    return research_chain, blog_chain, summary_chain, captions_chain

research_chain, blog_chain, summary_chain, captions_chain = get_pipeline()

topic = st.text_input("Enter a topic:", placeholder="e.g. Artificial Intelligence in India")

if st.button("🚀 Generate Content", type="primary"):
    if not topic:
        st.warning("Please enter a topic first!")
    else:
        with st.spinner("Step 1/4 — Researching..."):
            research = research_chain.invoke({"topic": topic})

        with st.spinner("Step 2/4 — Writing blog post..."):
            blog = blog_chain.invoke({"research": research})

        with st.spinner("Step 3/4 — Summarizing..."):
            summary = summary_chain.invoke({"blog": blog})

        with st.spinner("Step 4/4 — Creating captions..."):
            captions = captions_chain.invoke({"summary": summary})

        st.success("✅ Done!")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔍 Research")
            st.write(research)
            st.subheader("📝 Blog Post")
            st.write(blog)

        with col2:
            st.subheader("📌 Summary")
            st.write(summary)
            st.subheader("📱 Social Captions")
            st.write(captions)

        st.divider()
        st.download_button(
            label="⬇️ Download All Content",
            data=f"TOPIC: {topic}\n\nRESEARCH:\n{research}\n\nBLOG:\n{blog}\n\nSUMMARY:\n{summary}\n\nCAPTIONS:\n{captions}",
            file_name=f"{topic.replace(' ', '_')}_content.txt",
            mime="text/plain"
        )