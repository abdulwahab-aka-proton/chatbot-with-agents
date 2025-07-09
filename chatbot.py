from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from enum import Enum
from typing import TypedDict, List
from dotenv import load_dotenv
import re


load_dotenv()

llm = ChatGroq(model="your-model", temperature=0)

pdf_loader = PyPDFLoader("your-document.pdf")
documents = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="your-model",
    model_kwargs={"device": "cpu"},
)

vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

@tool
def calculate(expression: str) -> str:
    """Evaluate a basic math expression (+, -, *, /). Example: '5+5' or '10/2'"""
    try:
        if not re.match(r'^[\d\.\+\-\*\/\s]+$', expression):
            return "Error: Only basic math operations (+, -, *, /) with numbers are allowed"
        
        result = eval(expression)  
        return f"{result}"
    except ZeroDivisionError:
        return "Error: Division by zero"

math_tools = [calculate]
MATH_AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a math expert. When given a math expression:
    1. ALWAYS use the calculate tool
    2. The input should be the exact math expression like "5+5" or "10/2"
    3. Never try to calculate manually"""),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

math_agent = create_tool_calling_agent(
    llm=llm,
    tools=math_tools,
    prompt=MATH_AGENT_PROMPT
)

math_executor = AgentExecutor(
    agent=math_agent,
    tools=math_tools,
    verbose=True
)

class Route(str, Enum):
    DATA = "data"
    MATH = "math"
    UNKNOWN = "unknown"

class AgentState(TypedDict):
    question: str
    route: Route
    context: List[str]
    answer: str
    clarification: str

def classifier_node(state: AgentState) -> AgentState:
    user_input = state["question"]
    
    math_pattern = r"\d+\s*[\+\-\*\/]\s*\d+"

    if re.search(math_pattern, user_input):
        return {"route": Route.MATH}
    else:
        return {"route": Route.DATA}

def data_node(state: AgentState) -> AgentState:
    template = """Answer based on context:
        {context}
        
        Question: {question}
        Answer should be concise and to the point"""
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    answer = chain.invoke(state["question"])
    return {"answer": answer, "route": END}

def math_node(state: AgentState) -> AgentState:
    query = state["question"]
    clean_query = query.replace(" ", "")
    match = re.search(r"(\d+)([\+\-\*\/])(\d+)", clean_query)
    
    if not match:
        return {
            "route": END
        }
    
    result = math_executor.invoke({"input": query})
    return {"answer": result["output"], "route": END}

workflow = StateGraph(AgentState)
workflow.add_node("classifier", classifier_node)
workflow.add_node("data_agent", data_node)
workflow.add_node("math_agent", math_node)
workflow.add_conditional_edges(
    "classifier",
    lambda x: x["route"],
    {
        Route.DATA: "data_agent",
        Route.MATH: "math_agent"
    }
)
workflow.add_edge("math_agent", END)
workflow.add_edge("data_agent", END)
workflow.set_entry_point("classifier")
app = workflow.compile()

print ("Chatbot -- (type 'quit' to exit)")
while True:
    question = input("\033[93;1mYou: \033[0m")
    if question.lower() == "quit":
        break
    result = app.invoke({
            "question": question,
            "route": Route.UNKNOWN,
            "context": [],
            "answer": "",
            "clarification": ""
        })
    print (f"\n\033[92;1mBot:\033[0m {result['answer']}")
