import streamlit as st
from dotenv import load_dotenv
from newspaper import Article
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableSequence
import os

# --- HELPER FUNCTIONS ---

def fetch_article_content(url: str) -> str:
    """
    Fetches and parses the main content of a news article from a given URL.
    
    Args:
        url (str): The URL of the news article.
        
    Returns:
        str: The extracted text content of the article, or an error message.
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        st.error(f"Error fetching article from URL: {e}")
        return None

# --- LANGCHAIN IMPLEMENTATION WITH GEMINI ---

def get_analysis_chain() -> RunnableSequence:
    """
    Creates and returns a LangChain sequence for analyzing financial news using Google's Gemini model.

    This chain takes the article text and generates a structured report covering:
    - A concise summary
    - Sentiment analysis (Positive, Negative, Neutral) with justification
    - Key risks identified
    - Potential opportunities highlighted
    
    Returns:
        RunnableSequence: The configured LangChain analysis chain.
    """
    # Initialize the Gemini 1.5 Pro model for high-quality, nuanced analysis
    # The temperature is set to 0 to ensure deterministic and factual output
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

    # Define the structured prompt for the financial analysis
    template = """
    **Objective:** You are "FinBot," an expert financial analyst AI. Your task is to provide a clear, concise, and insightful analysis of the following financial news article. Your audience consists of investors and business stakeholders who need actionable information.

    **Instructions:**
    1.  Read the article text provided below carefully.
    2.  Generate a structured report with the following distinct sections: `### Executive Summary`, `### Sentiment Analysis`, `### Key Risks`, and `### Potential Opportunities`. Use markdown headings for each section.
    3.  **Executive Summary:** Provide a brief, neutral summary of the article's main points. What happened? Who are the key players? What is the core news?
    4.  **Sentiment Analysis:** Classify the overall sentiment of the news as `Positive`, `Negative`, or `Neutral`. Provide a one-sentence justification for your classification, citing specific information from the article.
    5.  **Key Risks:** Identify and list up to 3 potential risks for the involved companies, sectors, or the market based on the article. These should be specific and derived directly from the text.
    6.  **Potential Opportunities:** Identify and list up to 3 potential opportunities for investors or businesses based on the article. These should also be specific and directly supported by the text.

    **Article Text:**
    ```{article_text}```

    **Generated Report:**
    """

    prompt = PromptTemplate(
        input_variables=["article_text"],
        template=template
    )

    # Create the chain by linking the prompt with the LLM
    return prompt | llm

# --- STREAMLIT UI ---

def main():
    """
    The main function that sets up and runs the Streamlit web application.
    """
    # Load environment variables (for the Google API key)
    load_dotenv()

    # Configure the Streamlit page
    st.set_page_config(page_title="FinBot Financial Analyser", page_icon="🤖", layout="wide")

    # Header section
    st.title("📈 FinBot: Financial News Analyser & Report Generator")
    st.markdown("""
        Welcome to FinBot, This tool automates the analysis of financial news to provide you with actionable insights. 
        Simply enter the URL of a news article below to receive a structured report.
    """)
    st.divider()

    # Input section
    st.header("🔗 Analyze a News Article")
    url = st.text_input("Enter the URL of the financial news article:", placeholder="https://www.example.com/financial-news-article")

    # Analysis trigger
    if st.button("Generate Report", type="primary"):
        # Check for API Key
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("Google API key not found. Please set it in your .env file.")
            return
            
        if not url:
            st.warning("Please enter a URL to proceed.")
        else:
            with st.spinner("🔍 FinBot is reading and analyzing the article... Please wait."):
                # 1. Fetch article content
                article_text = fetch_article_content(url)
                
                if article_text:
                    # 2. Get the analysis chain
                    chain = get_analysis_chain()
                    
                    try:
                        # 3. Invoke the chain to get the report
                        result = chain.invoke({"article_text": article_text})
                        report_content = result.content
                        
                        # 4. Display the generated report
                        st.divider()
                        st.header("📊 Your Financial Analysis Report")
                        st.markdown(report_content)
                        st.success("Report generated successfully!")
                        
                    except Exception as e:
                        st.error(f"An error occurred during analysis: {e}")

    # Footer
    st.markdown("""
        <div style="text-align: center; margin-top: 50px; font-size: 12px; color: grey;">
            FinBot | Empowering Decisions with AI-Powered Market Intelligence
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()