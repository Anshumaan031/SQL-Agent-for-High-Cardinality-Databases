"""
Planner Agent
Analyzes user query to extract intent and entities
"""
import json
import re
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.state import AgentState


# Global LLM instance - will be set by main agent
_llm: ChatGoogleGenerativeAI = None


def set_planner_llm(api_key: str):
    """Initialize the LLM for the planner agent"""
    global _llm
    _llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0
    )


def planner_agent(state: AgentState) -> AgentState:
    """
    Planner Agent - Analyzes user query and plans approach.

    Extracts:
    - User's intent
    - Entities/values mentioned
    - Query complexity

    Args:
        state: Current agent state

    Returns:
        Updated state with intent and entities
    """
    user_query = state["user_query"]

    prompt = f"""You are a query planning agent for a SQL database system.

User query: "{user_query}"

Analyze this query and extract:
1. The user's intent (what they want to know)
2. Specific entities/values mentioned (names, locations, numbers, etc.)
3. Query complexity (simple, moderate, or complex)

Respond in JSON format:
{{
    "intent": "brief description of what user wants",
    "entities": ["list", "of", "entities", "mentioned"],
    "complexity": "simple|moderate|complex",
    "reasoning": "brief explanation of your analysis"
}}

Examples:
- Query: "Show me tracks by AC/DC"
  Response: {{"intent": "Find tracks by artist", "entities": ["AC/DC"], "complexity": "simple", "reasoning": "Simple query filtering tracks by artist name"}}

- Query: "What are the top 5 genres by total sales in the USA?"
  Response: {{"intent": "Find top genres by sales in USA", "entities": ["USA", "5"], "complexity": "moderate", "reasoning": "Requires aggregation and filtering by country"}}

Now analyze the user's query above."""

    response = _llm.invoke([{"role": "user", "content": prompt}])

    # Parse the JSON response
    try:
        # Extract JSON from response - handle list content
        content = response.content
        if isinstance(content, list):
            content = ' '.join(str(item) for item in content)
        elif not isinstance(content, str):
            content = str(content)
        content = content.strip()

        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            # Fallback parsing
            parsed = {
                "intent": user_query,
                "entities": [],
                "complexity": "simple",
                "reasoning": "Could not parse structured response"
            }

        intent = parsed.get("intent", user_query)
        entities = parsed.get("entities", [])
        reasoning = parsed.get("reasoning", "")

    except Exception as e:
        # Fallback if parsing fails
        intent = user_query
        entities = []
        reasoning = f"Parsing error: {str(e)}"

    # Create agent message
    agent_message = AIMessage(
        content=f"[PLANNER] Intent: {intent} | Entities: {entities} | Reasoning: {reasoning}",
        name="planner"
    )

    return {
        **state,
        "messages": [agent_message],
        "intent": intent,
        "entities": entities
    }
