from langchain.chat_models.gigachat import GigaChat

from langchain_community.document_loaders import *
from langchain_community.tools.tavily_search import TavilySearchResults


from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

from langchain.agents import AgentExecutor, create_gigachat_functions_agent, load_tools


from typing import List


def create_gigachat_agent_prompt_() -> ChatPromptTemplate:
    messages = [
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
    return ChatPromptTemplate.from_messages(messages=messages)


def create_agent(
    llm: GigaChat,
    tools: list,
    system_prompt: str,
) -> str:
    """Create a function-calling agent and add it to the graph."""
    system_prompt += "\nРаботай автономно, следуя своей специализации, используя инструменты которые тебе даны"
    " Не проси никаких пояснений. Очень важно чтобы ты использовал один из инструментов которые тебе даны."
    " Твои коллеги из твоей команды (и других команд) будут сотрудничать с тобой согласно их специализации."
    " Ты был выбран не просто так! Ты - один из участников следующей команды: {team_members}"
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_gigachat_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor


def agent_node(state, agent, name):
    print(name)
    result = agent.invoke(state)
    print(name)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


def create_team_supervisor(llm: GigaChat, system_prompt, members) -> str:
    """An LLM-based router."""
    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Выбери следующую роль.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "type": "string",
                    "enum": options,
                    "description": "Название следующего участника",
                },
            },
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    ).partial(options=str(options), team_members=", ".join(members))
    return (
        prompt
        | llm.bind_tools(tools=[function_def], tool_choice="route")
        | JsonOutputFunctionsParser()
    )


tavily_tool = TavilySearchResults(max_results=5)


@tool
def scrape_webpages(urls: List[str]) -> str:
    """
    Использует парсинг для выскабливания информации из ссылок urls для детализированной информации

    Args:
        urls List[str]: Ссылки для парсинга

    Returns:
        str: Текст веб страниц.
    """
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return "\n\n".join(
        [
            f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )


arxiv_tools = load_tools(
    ["arxiv"],
)
