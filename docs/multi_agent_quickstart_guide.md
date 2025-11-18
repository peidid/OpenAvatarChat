# Multi-Agent Advisor Chatbot - Quick Start Guide

This guide will help you get started building your multi-agent advising chatbot system with minimal setup time.

---

## Overview

You'll build a system where:
1. Students ask questions via voice/text
2. An orchestrator routes queries to specialized department advisors
3. Each advisor has access to course catalogs and FAQs
4. Conversations are stored and can be resumed
5. The system integrates with your existing OpenAvatarChat infrastructure

**Expected Timeline:** 2-3 weeks for MVP

---

## Prerequisites

### 1. Required Accounts & API Keys

```bash
# OpenAI API (for LLM and embeddings)
export OPENAI_API_KEY="sk-..."

# Optional: LangSmith for monitoring
export LANGCHAIN_API_KEY="ls__..."
export LANGCHAIN_TRACING_V2=true
```

### 2. Install Dependencies

```bash
# Add to your pyproject.toml
[tool.uv.dependencies]
langgraph = "^0.2.0"
langchain-openai = "^0.1.0"
langchain-core = "^0.2.0"
chromadb = "^0.5.0"
psycopg2-binary = "^2.9.0"
redis = "^5.0.0"

# Install
uv sync
```

### 3. Set Up Databases

**PostgreSQL:**
```bash
# Install PostgreSQL
sudo apt-get install postgresql

# Create database
createdb advisor_db

# Run schema creation script (see below)
psql advisor_db < setup_database.sql
```

**Redis:**
```bash
# Install Redis
sudo apt-get install redis-server

# Start Redis
redis-server
```

**ChromaDB:**
```bash
# ChromaDB runs in-process, no separate installation needed
# Just specify a directory in config
```

---

## Step-by-Step Setup

### Step 1: Database Schema (5 minutes)

Create `scripts/setup_database.sql`:

```sql
-- Students table
CREATE TABLE students (
    student_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(200),
    email VARCHAR(200),
    major VARCHAR(100),
    year INTEGER,
    gpa DECIMAL(3,2),
    enrolled_courses TEXT[],
    completed_courses JSONB,
    interests TEXT[],
    created_at TIMESTAMP DEFAULT NOW()
);

-- Chat sessions
CREATE TABLE chat_sessions (
    session_id UUID PRIMARY KEY,
    student_id VARCHAR(50) REFERENCES students(student_id),
    department VARCHAR(50),
    start_time TIMESTAMP DEFAULT NOW(),
    end_time TIMESTAMP,
    message_count INTEGER DEFAULT 0,
    topics TEXT[],
    summary TEXT,
    status VARCHAR(20) DEFAULT 'active'
);

-- Messages
CREATE TABLE messages (
    message_id SERIAL PRIMARY KEY,
    session_id UUID REFERENCES chat_sessions(session_id),
    student_id VARCHAR(50) REFERENCES students(student_id),
    agent_name VARCHAR(100),
    role VARCHAR(20),
    content TEXT,
    metadata JSONB,
    timestamp TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_messages_session ON messages(session_id);
CREATE INDEX idx_messages_student ON messages(student_id);
CREATE INDEX idx_messages_timestamp ON messages(timestamp DESC);

-- Courses
CREATE TABLE courses (
    course_id VARCHAR(20) PRIMARY KEY,
    department VARCHAR(50),
    course_name VARCHAR(200),
    description TEXT,
    credits INTEGER DEFAULT 3,
    prerequisites TEXT[],
    offered_terms TEXT[],
    capacity INTEGER,
    current_enrollment INTEGER DEFAULT 0
);

-- Insert sample courses
INSERT INTO courses VALUES
('CS101', 'cs', 'Introduction to Programming', 'Learn Python programming basics', 3, ARRAY[]::TEXT[], ARRAY['Fall', 'Spring'], 150, 120),
('CS201', 'cs', 'Data Structures', 'Arrays, linked lists, trees, graphs', 3, ARRAY['CS101'], ARRAY['Fall', 'Spring'], 100, 85),
('CS301', 'cs', 'Algorithms', 'Algorithm design and analysis', 3, ARRAY['CS201', 'MATH220'], ARRAY['Fall', 'Spring'], 80, 65),
('CS401', 'cs', 'Machine Learning', 'Introduction to ML algorithms', 3, ARRAY['CS301', 'MATH220'], ARRAY['Spring'], 60, 45),
('IS101', 'is', 'Information Systems Fundamentals', 'Overview of IS field', 3, ARRAY[]::TEXT[], ARRAY['Fall', 'Spring'], 120, 95),
('IS201', 'is', 'Database Systems', 'SQL and database design', 3, ARRAY['IS101'], ARRAY['Fall', 'Spring'], 100, 80);

-- Insert sample student
INSERT INTO students VALUES
('STU001', 'John Doe', 'john.doe@university.edu', 'Computer Science', 3, 3.75, 
 ARRAY['CS301', 'MATH250'], 
 '[{"course_id": "CS101", "grade": "A", "semester": "Fall2022", "credits": 3}, 
   {"course_id": "CS201", "grade": "A-", "semester": "Spring2023", "credits": 3}]'::jsonb,
 ARRAY['AI/ML', 'Web Development']
);
```

Run it:
```bash
psql advisor_db < scripts/setup_database.sql
```

### Step 2: Load Knowledge Base (10 minutes)

Create `scripts/load_knowledge_base.py`:

```python
import chromadb
from chromadb.utils import embedding_functions
import os

def load_knowledge_base():
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path="./data/chroma_db")
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    
    # CS Department Knowledge
    cs_courses = client.create_collection(
        name="cs_courses",
        embedding_function=embedding_fn
    )
    
    cs_courses.add(
        documents=[
            "CS101 Introduction to Programming: Learn Python programming basics. No prerequisites. Offered Fall and Spring.",
            "CS201 Data Structures: Study of arrays, linked lists, trees, and graphs. Prerequisites: CS101. Offered Fall and Spring.",
            "CS301 Algorithms: Algorithm design and analysis including sorting, searching, and graph algorithms. Prerequisites: CS201, MATH220. Offered Fall and Spring.",
            "CS401 Machine Learning: Introduction to supervised and unsupervised learning, neural networks. Prerequisites: CS301, MATH220. Offered Spring only.",
        ],
        metadatas=[
            {"course_code": "CS101", "course_name": "Introduction to Programming", "credits": 3},
            {"course_code": "CS201", "course_name": "Data Structures", "credits": 3},
            {"course_code": "CS301", "course_name": "Algorithms", "credits": 3},
            {"course_code": "CS401", "course_name": "Machine Learning", "credits": 3},
        ],
        ids=["CS101", "CS201", "CS301", "CS401"]
    )
    
    # CS FAQs
    cs_faqs = client.create_collection(
        name="cs_faqs",
        embedding_function=embedding_fn
    )
    
    cs_faqs.add(
        documents=[
            "CS majors must complete 120 total credits, including 45 credits in CS courses. Core requirements include CS101, CS201, CS301, and 5 elective courses.",
            "To declare a CS major, you must complete CS101 and CS201 with a grade of B- or better and have an overall GPA of 2.5 or higher.",
            "The AI/ML specialization requires CS401 Machine Learning, CS402 Deep Learning, CS403 Natural Language Processing, plus 2 electives from an approved list.",
            "Most CS courses have programming assignments due weekly. Exams are typically midterm and final. Group projects are common in 300/400 level courses.",
        ],
        metadatas=[
            {"question": "What are CS degree requirements?", "category": "requirements"},
            {"question": "How do I declare a CS major?", "category": "major_declaration"},
            {"question": "What is the AI/ML specialization?", "category": "specialization"},
            {"question": "What is the workload like for CS courses?", "category": "general"},
        ],
        ids=["cs_faq_1", "cs_faq_2", "cs_faq_3", "cs_faq_4"]
    )
    
    # IS Department Knowledge
    is_courses = client.create_collection(
        name="is_courses",
        embedding_function=embedding_fn
    )
    
    is_courses.add(
        documents=[
            "IS101 Information Systems Fundamentals: Overview of information systems in business. No prerequisites. Offered Fall and Spring.",
            "IS201 Database Systems: SQL, database design, and data modeling. Prerequisites: IS101. Offered Fall and Spring.",
        ],
        metadatas=[
            {"course_code": "IS101", "course_name": "Information Systems Fundamentals", "credits": 3},
            {"course_code": "IS201", "course_name": "Database Systems", "credits": 3},
        ],
        ids=["IS101", "IS201"]
    )
    
    is_faqs = client.create_collection(
        name="is_faqs",
        embedding_function=embedding_fn
    )
    
    is_faqs.add(
        documents=[
            "IS majors focus on the intersection of business and technology. The program requires 42 IS credits plus 18 business credits.",
            "IS students often pursue careers in business analysis, data analytics, IT consulting, and systems management.",
        ],
        metadatas=[
            {"question": "What is IS?", "category": "general"},
            {"question": "What careers can IS majors pursue?", "category": "career"},
        ],
        ids=["is_faq_1", "is_faq_2"]
    )
    
    # BS Department (add similar collections)
    bs_courses = client.create_collection(
        name="bs_courses",
        embedding_function=embedding_fn
    )
    
    bs_faqs = client.create_collection(
        name="bs_faqs",
        embedding_function=embedding_fn
    )
    
    # General advisor knowledge
    general_courses = client.create_collection(
        name="general_courses",
        embedding_function=embedding_fn
    )
    
    general_faqs = client.create_collection(
        name="general_faqs",
        embedding_function=embedding_fn
    )
    
    general_faqs.add(
        documents=[
            "Undergraduate students must complete 120 credits to graduate, including general education requirements, major requirements, and electives.",
            "Add/drop period is the first week of classes. After that, students need instructor approval to add courses.",
            "Financial aid applications are due by March 1st for the following academic year. Submit FAFSA and university aid application.",
        ],
        metadatas=[
            {"question": "How many credits needed to graduate?", "category": "graduation"},
            {"question": "When is add/drop period?", "category": "registration"},
            {"question": "How do I apply for financial aid?", "category": "financial_aid"},
        ],
        ids=["gen_faq_1", "gen_faq_2", "gen_faq_3"]
    )
    
    print("✅ Knowledge base loaded successfully!")

if __name__ == "__main__":
    load_knowledge_base()
```

Run it:
```bash
python scripts/load_knowledge_base.py
```

### Step 3: Minimal Working Example (30 minutes)

Create `examples/minimal_advisor.py`:

```python
"""
Minimal multi-agent advisor example
Run: python examples/minimal_advisor.py
"""

import os
from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import chromadb
from chromadb.utils import embedding_functions

# Set API key
os.environ["OPENAI_API_KEY"] = "your-key-here"

# Simple state
class State(TypedDict):
    messages: list
    department: str
    student_query: str

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

# LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Orchestrator node
def orchestrator(state: State) -> State:
    """Route to appropriate department"""
    query = state["messages"][-1].content
    
    # Simple routing with structured output
    class RouteDecision(BaseModel):
        department: Literal["cs", "is", "bs", "general"]
        reasoning: str
    
    prompt = f"""Route this student query to the appropriate department:
    Query: {query}
    
    Departments:
    - cs: Computer Science (programming, algorithms, AI/ML)
    - is: Information Systems (business technology, databases)
    - bs: Business School (management, finance)
    - general: General advising (registration, financial aid)
    """
    
    structured_llm = llm.with_structured_output(RouteDecision)
    decision = structured_llm.invoke(prompt)
    
    state["department"] = decision.department
    state["student_query"] = query
    
    print(f"\n🔀 Routing to: {decision.department}")
    print(f"   Reasoning: {decision.reasoning}\n")
    
    return state

# Department advisor node
def department_advisor(state: State) -> State:
    """Provide department-specific advice"""
    dept = state["department"]
    query = state["student_query"]
    
    # Retrieve relevant knowledge
    courses_collection = chroma_client.get_collection(
        name=f"{dept}_courses",
        embedding_function=embedding_fn
    )
    
    faqs_collection = chroma_client.get_collection(
        name=f"{dept}_faqs",
        embedding_function=embedding_fn
    )
    
    # Search
    courses = courses_collection.query(query_texts=[query], n_results=2)
    faqs = faqs_collection.query(query_texts=[query], n_results=2)
    
    # Format context
    context = "Relevant Courses:\n"
    if courses['documents'] and courses['documents'][0]:
        for doc in courses['documents'][0]:
            context += f"- {doc}\n"
    
    context += "\nRelevant FAQs:\n"
    if faqs['documents'] and faqs['documents'][0]:
        for doc in faqs['documents'][0]:
            context += f"- {doc}\n"
    
    # Generate response
    prompt = f"""You are a {dept.upper()} academic advisor.
    
Context:
{context}

Student Question: {query}

Provide helpful, specific advice. Reference actual courses when relevant.
"""
    
    response = llm.invoke(prompt)
    
    state["messages"].append(AIMessage(content=response.content))
    
    print(f"🤖 {dept.upper()} Advisor: {response.content}\n")
    
    return state

# Build graph
def build_graph():
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("orchestrator", orchestrator)
    workflow.add_node("cs_advisor", lambda s: department_advisor({**s, "department": "cs"}))
    workflow.add_node("is_advisor", lambda s: department_advisor({**s, "department": "is"}))
    workflow.add_node("bs_advisor", lambda s: department_advisor({**s, "department": "bs"}))
    workflow.add_node("general_advisor", lambda s: department_advisor({**s, "department": "general"}))
    
    # Routing logic
    def route(state: State):
        return f"{state['department']}_advisor"
    
    # Set entry
    workflow.set_entry_point("orchestrator")
    
    # Add edges
    workflow.add_conditional_edges(
        "orchestrator",
        route,
        {
            "cs_advisor": "cs_advisor",
            "is_advisor": "is_advisor",
            "bs_advisor": "bs_advisor",
            "general_advisor": "general_advisor"
        }
    )
    
    # End nodes
    workflow.add_edge("cs_advisor", END)
    workflow.add_edge("is_advisor", END)
    workflow.add_edge("bs_advisor", END)
    workflow.add_edge("general_advisor", END)
    
    return workflow.compile()

# Main
def main():
    print("🎓 Multi-Agent Academic Advisor")
    print("=" * 50)
    
    graph = build_graph()
    
    # Test queries
    test_queries = [
        "What are the prerequisites for machine learning?",
        "How do I declare an IS major?",
        "When is the add/drop period?",
        "What courses should I take for AI specialization?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"📝 Student: {query}")
        
        state = {
            "messages": [HumanMessage(content=query)],
            "department": "",
            "student_query": ""
        }
        
        result = graph.invoke(state)
        
        print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
```

Run it:
```bash
# Make sure you set your API key first!
export OPENAI_API_KEY="sk-..."

python examples/minimal_advisor.py
```

**Expected Output:**
```
🎓 Multi-Agent Academic Advisor
==================================================

==================================================
📝 Student: What are the prerequisites for machine learning?

🔀 Routing to: cs
   Reasoning: Query about ML prerequisites is CS-related

🤖 CS Advisor: CS401 Machine Learning requires CS301 Algorithms and MATH220 Linear Algebra as prerequisites. You should complete these courses before enrolling in machine learning. CS301 itself requires CS201 Data Structures, so make sure you've completed that sequence...

==================================================
```

### Step 4: Integrate with OpenAvatarChat (1 hour)

Create the handler file structure:

```bash
mkdir -p src/handlers/llm/multi_agent
touch src/handlers/llm/multi_agent/__init__.py
```

Copy the files from the implementation examples:
- `multi_agent_handler.py` (main handler)
- `graph_builder.py` (agent graph)
- `knowledge_base.py` (data access)
- `state.py` (state definition)
- `tools.py` (advisor tools)

Update your config:

```yaml
# config/chat_with_multi_agent_advisor.yaml

default:
  chat_engine:
    model_root: "models"
    handler_configs:
      
      # Multi-agent advisor
      MultiAgentAdvisor:
        enabled: True
        module: llm/multi_agent/multi_agent_handler
        openai_api_key: ${OPENAI_API_KEY}
        postgres_url: "postgresql://localhost/advisor_db"
        vector_db_path: "./data/chroma_db"
        departments: ["cs", "is", "bs"]
      
      # Existing handlers
      SileraVad:
        enabled: True
        module: vad/silerovad/vad_handler/silero
      
      SenseVoice:
        enabled: True
        module: asr/sensevoice/asr_handler_sensevoice
      
      CosyVoiceBailian:
        enabled: True
        module: tts/bailian_tts/tts_handler_cosyvoice_bailian
        api_key: ${DASHSCOPE_API_KEY}
      
      LiteAvatar:
        enabled: True
        module: avatar/liteavatar/avatar_handler_liteavatar
        avatar_name: "20250408/sample_data"
        fps: 25
        use_gpu: true
```

Run it:
```bash
uv run src/demo.py --config config/chat_with_multi_agent_advisor.yaml
```

---

## Testing Your System

### Test Queries

Try these queries to validate each agent:

**CS Advisor:**
- "What are the prerequisites for machine learning?"
- "I want to specialize in AI. What courses should I take?"
- "Is CS301 offered in the Spring?"
- "How hard is the CS program?"

**IS Advisor:**
- "What's the difference between CS and IS?"
- "How do I declare an IS major?"
- "What careers can IS graduates pursue?"
- "Do I need programming for IS?"

**General Advisor:**
- "How many credits do I need to graduate?"
- "When is the add/drop period?"
- "How do I apply for financial aid?"
- "Can I change my major?"

### Check Logs

```bash
# Watch for routing decisions
tail -f logs/advisor.log | grep "Routing to"

# Check database
psql advisor_db -c "SELECT count(*) FROM messages;"
psql advisor_db -c "SELECT agent_name, count(*) FROM messages GROUP BY agent_name;"

# Check Redis cache
redis-cli keys "session:*"
```

---

## Common Issues & Solutions

### Issue 1: "OpenAI API Error"
```bash
# Check API key
echo $OPENAI_API_KEY

# Test directly
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Issue 2: "Cannot connect to PostgreSQL"
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Test connection
psql advisor_db -c "SELECT version();"

# Check connection string
psql "postgresql://localhost/advisor_db"
```

### Issue 3: "ChromaDB not found"
```bash
# Verify ChromaDB installation
python -c "import chromadb; print(chromadb.__version__)"

# Check data directory exists
ls -la data/chroma_db/
```

### Issue 4: "Agent routing is incorrect"
```bash
# Enable debug logging
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="your-langsmith-key"

# Adjust routing prompt in orchestrator
# Increase confidence threshold
# Add more examples
```

---

## Next Steps

### Week 1: MVP
- ✅ Set up databases
- ✅ Load sample data
- ✅ Test minimal example
- ✅ Integrate with OpenAvatarChat

### Week 2: Enhancement
- [ ] Add more departments
- [ ] Expand knowledge base
- [ ] Implement student authentication
- [ ] Add conversation summarization
- [ ] Create admin dashboard

### Week 3: Production
- [ ] Load testing
- [ ] Security audit
- [ ] Monitoring setup
- [ ] User acceptance testing
- [ ] Documentation

---

## Cost Estimation

**For 1000 students with 10 conversations/month:**

| Service | Usage | Cost/Month |
|---------|-------|------------|
| OpenAI GPT-4 | ~500K tokens | $15-25 |
| OpenAI Embeddings | ~2M tokens | $0.20 |
| PostgreSQL | Small instance | $0 (local) or $25 (cloud) |
| Redis | 1GB | $0 (local) or $10 (cloud) |
| ChromaDB | Self-hosted | $0 |
| **Total** | | **$15-60/month** |

**Cost Optimization Tips:**
- Cache common questions (FAQ hits)
- Use GPT-3.5-turbo for simple routing
- Batch embeddings
- Set token limits

---

## Support & Resources

**Documentation:**
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [OpenAI API Docs](https://platform.openai.com/docs)
- [ChromaDB Docs](https://docs.trychroma.com/)

**Community:**
- [LangChain Discord](https://discord.gg/langchain)
- [OpenAvatarChat Issues](https://github.com/HumanAIGC-Engineering/OpenAvatarChat/issues)

**Monitoring:**
- [LangSmith](https://smith.langchain.com/) - Agent tracing & debugging
- [LangFuse](https://langfuse.com/) - Open-source LLM observability

---

## Conclusion

You now have a complete setup guide for building a multi-agent advising chatbot! 

**Your MVP should be able to:**
1. ✅ Route student queries to appropriate advisors
2. ✅ Answer questions using department knowledge
3. ✅ Check course prerequisites
4. ✅ Store conversation history
5. ✅ Work with your existing avatar system

**Remember:**
- Start simple (2-3 agents)
- Test with real students early
- Monitor costs and performance
- Iterate based on feedback

Good luck! 🚀
