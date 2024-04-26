from dotenv import load_dotenv
import telebot
from database import create_database, save_file_info, get_user_files
from typing import NoReturn
import os
from my_class import DataAbstraction
from langchain.chat_models.gigachat import GigaChat

from langchain_community.embeddings.gigachat import GigaChatEmbeddings
from langchain_community.document_loaders import *

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.output_parsers import PydanticOutputParser

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from typing import Dict, Optional

from langchain_experimental.utilities import PythonREPL
from typing_extensions import TypedDict


from langgraph.graph import StateGraph, END


from typing import Annotated, List


import functools

import operator

from help_functions import *

# Наш собственный агент по анализу данных (в репозитории base.py)
from base import create_pandas_dataframe_agent

create_database()
load_dotenv()

try:
    os.mkdir("files")
except:
    pass
bot = telebot.TeleBot(os.getenv("token"))

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "ls__0808bff299b84be3bafb44257673fd26"
os.environ["LANGCHAIN_PROJECT"] = "Multi-agent Collaboration"
os.environ["TAVILY_API_KEY"] = os.getenv("tavile")
gigachat_creds = os.getenv("credentials")
use_pro = True

# INITIALIZING AGENTS

if use_pro:
    llm = GigaChat(
        credentials=gigachat_creds,
        verify_ssl_certs=False,
        stream=True,
        model="GigaChat-Pro-preview",
        scope="GIGACHAT_API_CORP",
    )
else:
    llm = GigaChat(
        credentials=gigachat_creds,
        verify_ssl_certs=False,
        stream=True,
        model="GigaChat-preview",
        scope="GIGACHAT_API_CORP",
    )


text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
embeddings = GigaChatEmbeddings(credentials=gigachat_creds, verify_ssl_certs=False)


class ResearchTeamState(TypedDict):
    # A message is added after each team member finishes
    messages: Annotated[List[BaseMessage], operator.add]
    # The team members are tracked so they are aware of
    # the others' skill-sets
    team_members: List[str]
    # Used to route work. The supervisor calls a function
    # that will update this every time it makes a decision
    next: str


search_agent = create_agent(
    llm,
    [tavily_tool],
    "Ты ассистент научного исследователя, который может производить поиск свежей информации используя поисковую систему tavily",
)
search_node = functools.partial(agent_node, agent=search_agent, name="Search")

research_agent = create_agent(
    llm,
    [scrape_webpages],
    "Ты ассистент научного исследователя, который может считывать текст с urls для получения детализированной информации используя scrape_webpages функцию",
)
research_node = functools.partial(agent_node, agent=research_agent, name="Web Scraper")

arxiv_agent = create_agent(
    llm,
    arxiv_tools,
    "Ты ассистент научного исследователя, искать информацию по какой либо теме среди архива научных статей, используя инструмент arxiv.",
)
arxiv_node = functools.partial(agent_node, agent=arxiv_agent, name="Arxiv Browser")


supervisor_agent = create_team_supervisor(
    llm,
    "Ты - менеджер от которого требуется провести сообщение между"
    " следующими работниками: Search, Web Scraper, Arxiv Browser. Учитывая полученный запрос"
    " ответь названием работника который должен действовать следующим. Каждый работник"
    " выполняет свою задачу и ответает на запрос полученным результатом и статусом. Если запрос уже является завершенным,"
    " ответь словом FINISH."
    "Учитывая приведенное ниже обсуждение, кто должен действовать следующим?"
    " Или же мы должны FINISH? Выбери одно из: {options}",
    ["Search", "Web Scraper", "Arxiv Browser"],
)

research_graph = StateGraph(ResearchTeamState)
research_graph.add_node("Search", search_node)
research_graph.add_node("Web Scraper", research_node)
research_graph.add_node("Arxiv Browser", arxiv_node)
research_graph.add_node("supervisor", supervisor_agent)

# Define the control flow
research_graph.add_edge("Search", "supervisor")
research_graph.add_edge("Web Scraper", "supervisor")
research_graph.add_edge("Arxiv Browser", "supervisor")
research_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "Search": "Search",
        "Web Scraper": "Web Scraper",
        "Arxiv Browser": "Arxiv Browser",
        "FINISH": END,
    },
)


research_graph.set_entry_point("supervisor")
chain = research_graph.compile()


# The following functions interoperate between the top level graph state
# and the state of the research sub-graph
# this makes it so that the states of each graph don't get intermixed
def enter_chain(message: str):
    results = {
        "messages": [HumanMessage(content=message)],
    }
    return results


research_chain = enter_chain | chain


@tool
def research_tool(message: str) -> str:
    """
    Инструмент для поиска информации в интернете и в архивах научных статей.

    Args:
        message (str): Вопрос на который нужно найти ответ в интернете

    Returns:
        str: Сжатая информация из интернета.
    """
    research_tool_chain = research_chain
    answer = research_chain.invoke(message, {"recursion_limit": 10})["messages"][-1]
    return StrOutputParser().invoke(answer)


prompt_ = PromptTemplate(
    template="""
    Ты - проверяющий, который должен определить, достоверен ли представленный, факт/гипотеза/идея и соответствует ли он действительности.
    Основано ли утверждение на фактах? Справедливо ли оно с точки зрения логики? Не нарушает ли оно научных принципов?
    Дай бинарную оценку 'yes' или 'no' оценку которая выражает обоснован ли ответ и поддержан ли он фактами.
    Предоставь только оценку 'yes' или 'no' и не предоставляй никаких обьяснений.

    Далее представлено утверждение на вопрос: {input}

    assistant:
    """,
    input_variables=["input"],
)

check_chain = prompt_ | llm | StrOutputParser()


@tool
def check_statement(statement: str) -> str:
    """
    Проверяет идею/гипотезу/утверждение на достоверность и логичность.

    Args:
        statement (str): утверджение для проверки

    Returns:
        str: является ли утверждение достоверным
    """

    return check_chain.invoke({"input": statement})


prompt_ = PromptTemplate.from_template(
    """
    Ты - ассистент научного исследования и твоя задача придумать список гипотез
    связанных с объектом исследования, с целью повышения метрик. Тебе даны:

    Обьект исследования: {research_object}

    Описание объекта: {object_description}

    Метрики: {metrics}

    Тебе нужно придумать хотя бы 3 гипотезы которые будут разной природы, которые должны повысить метрику(метрики)

    Для каждой гипотезы укажи следующие аспекты:

    Формулировка гипотезы: математическая модель или просто идея

    Сложность реализации гипотезы: Выражение сложности реализации экспериментов в денежном эквиваленте или человекочасах.

    Потенциальный прирост метрики: Это может быть число, или примерная оценка.

    Далее напиши этот список гипотез
    assistant:
    """
)
hypothesis_generating_chain = prompt_ | llm | StrOutputParser()


@tool
def generate_hypothesis(
    research_object: str, object_description: str, metrics: str
) -> str:
    """
    Инструмент который генерирует список гипотез который могут повысить метрику, связанную с объектом исследования.
    Для каждой гипотезы прилагает краткое описание.

    Args:
      research_object (str): Название объекта исследования(это может быть процесс, проблема, или какой то предмет)
      object_description (str): Описание объекта исследования, можно указать уточнения насчет природы объекта и научной сферы.
      metrics (str): Метрики которые связаны с объектом исследования, это может быть что то составное, например количество прибыли, или что то легко выражаемое математической формулой, например точность угадывания модели.
    Returns:
      str: Список гипотез которые могут повысить метрику, указывать на стороны проблемы, а также описания к ним, например количество ресурсов на реализацию гипотезы, или потенциальная прибыль.
    """
    result = hypothesis_generating_chain.invoke(
        {
            "research_object": research_object,
            "object_description": object_description,
            "metrics": metrics,
        }
    )
    return result


prompt_template = """
Ты - ассистент научного исследователя и тебе представлена проблематика по которой нужно строго определить объект исследования для человека

Строго соблюдай следующие инструкции:
{format_instructions}

Получи следующую информацию из запроса пользователя:

problem: {problem}

И дополнительной информации:

problem_context: {problem_context}

Далее твой ответ:
assistant:
"""


class ResearchObject(BaseModel):
    research_object: Optional[str] = Field(
        description="Извлеките название объекта исследования из текста"
    )
    object_description: Optional[str] = Field(
        description="Создайте описание объекта исследования из текста"
    )
    metrics: Optional[str] = Field(
        description="Напишите метрики которые соответствуют проблеме"
    )


object_output_parser = PydanticOutputParser(pydantic_object=ResearchObject)
object_instructions = object_output_parser.get_format_instructions()
prompt_object_parser = PromptTemplate.from_template(template=prompt_template)


def get_object_fields(problem, problem_context, instructions, parser):
    prompt_value = prompt_object_parser.format_prompt(
        problem=problem,
        problem_context=problem_context,
        format_instructions=instructions,
    )
    chain = prompt_object_parser | llm | parser
    response = chain.invoke(
        {
            "problem": problem,
            "problem_context": problem_context,
            "format_instructions": instructions,
        }
    )
    return response.dict()


@tool
def analyze_problematic(problem: str, problem_context: str) -> Dict[str, str]:
    """
    Инструмент который собирает информацию о проблеме и формирулирует объект исследования.
    Он позволяет на основе постановки проблемы и контекста задачи(похожие статьи, информация из интернета) проанализировать предметную область
    и выделить объект исследования.
    Args:
      problem (str): Описание проблемы, задачи
      problem_context (str): Дополнительные материалы для понимания задачи
    Returns:
      Dict: {'research_object' : Название объекта исследования,
             'object_description': Описание объекта исследования,
             'metrics' : Метрики которые связаны с объектом исследования.}
    """
    return get_object_fields(
        problem, problem_context, object_instructions, object_output_parser
    )


answer_agent = create_agent(
    llm,
    [check_statement, research_tool],
    "Ты ассистент научного исследователя, который отвечает за общение с пользователем и который должен наталкивать пользователя на принятие нужных решений и дать ему самому дойти до нужной идеи."
    " Ты получаешь какое то сообщение и если в нем содержится какой то факт или гипотеза то ты можешь проверить ее с помощью инструмента check_statement"
    " Далее ты должен либо указать на ошибку в сообщении, задав наводящий вопрос который натолкнет человека на эту ошибку, либо задать вопрос который продвинет идею дальше."
    " От тебя требуется ответить на сообщение пользователя данным способом. Для поиска дополнительной информации в интернете ты можешь воспользоваться инструментом research_tool."
    " Данный инструмент позволит тебе обратиться к поиску в интернете если для тебя информация о вопросе недоступна"
    " Ты отвечаешь за общение с человеком.",
)
answer_node = functools.partial(agent_node, agent=answer_agent, name="Answer Manager")

problem_agent = create_agent(
    llm,
    [analyze_problematic, generate_hypothesis, research_tool],
    "Ты ассистент научного исследователя, который должен положить начало исследованию. Ты получаешь проблему пользователя, и должен отвечает за общение с пользователем и который должен наталкивать пользователя на принятие нужных решений и дать ему самому дойти до нужной идеи."
    " Ты получаешь какое то сообщение и если в нем содержится какой то факт или гипотеза то ты можешь проверить ее с помощью инструмента check_statement"
    " Далее ты должен либо указать на ошибку в сообщении, задав наводящий вопрос который натолкнет человека на эту ошибку, либо задать вопрос который продвинет идею дальше."
    " От тебя требуется ответить на сообщение пользователя данным способом. Для поиска дополнительной информации ты можешь воспользоваться инструментом research_tool."
    " Ты отвечаешь за общение с человеком.",
)
problem_node = functools.partial(agent_node, agent=answer_agent, name="Problem Manager")

conversate_supervisor_agent = create_team_supervisor(
    llm,
    "Ты - ассистент научного исследователя который должен принять вопрос/сообщение/проблему/задачу пользователя и распределить её решение "
    " между следующими работниками: Answer Manager, Problem Manager. Учитывая полученный запрос"
    " ответь названием работника который должен действовать следующим. Каждый работник"
    " выполняет свою задачу и ответает на запрос полученным результатом и статусом."
    " Problem Manager - работник который берет изначальную проблему пользователя и проводит исследование для её решения. После его ответа можно отдавать результат сразу пользователю, сказав FINISH"
    " Answer Manager - работник который берет вопрос или утверждение пользователя и генерирует ответ с учетом дополнительной информации. Работник ответит наводящим вопросом который поможет пользователю дойти до нужной идеи."
    " Если запрос уже является завершенным,ответь словом FINISH."
    "Учитывая приведенное ниже обсуждение, кто должен действовать следующим?"
    " Или же мы должны FINISH? Выбери одно из: {options}",
    ["Answer Manager", "Problem Manager"],
)


class AnswerTeamState(TypedDict):
    # A message is added after each team member finishes
    messages: Annotated[List[BaseMessage], operator.add]
    # The team members are tracked so they are aware of
    # the others' skill-sets
    team_members: List[str]
    # Used to route work. The supervisor calls a function
    # that will update this every time it makes a decision
    next: str


answer_graph = StateGraph(AnswerTeamState)
answer_graph.add_node("Answer Manager", answer_node)
answer_graph.add_node("Problem Manager", problem_node)
answer_graph.add_node("supervisor", conversate_supervisor_agent)

# Define the control flow
answer_graph.add_edge("Answer Manager", "supervisor")
answer_graph.add_edge("Problem Manager", "supervisor")
answer_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "Answer Manager": "Answer Manager",
        "Problem Manager": "Problem Manager",
        "FINISH": END,
    },
)


answer_graph.set_entry_point("supervisor")
chain = answer_graph.compile()


# The following functions interoperate between the top level graph state
# and the state of the research sub-graph
# this makes it so that the states of each graph don't get intermixed
def enter_chain(message: str):
    results = {
        "messages": [HumanMessage(content=message)],
    }
    return results


answer_chain = enter_chain | chain


def collect_dfs(directory_path):
    collectus = []
    try:
        for i in os.listdir(directory_path):
            datum = DataAbstraction(os.path.join(directory_path, i))
            if datum.sql_like:
                collectus.append(datum.get_df)
    except:
        pass
    return collectus


class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str
    current_files: str


class ActionTeamState(TypedDict):
    # A message is added after each team member finishes
    messages: Annotated[List[BaseMessage], operator.add]
    # The team members are tracked so they are aware of
    # the others' skill-sets
    team_members: List[str]
    # Used to route work. The supervisor calls a function
    # that will update this every time it makes a decision
    next: str
    current_files: str


def get_action_last_message(state: ActionTeamState) -> str:
    return state["messages"][-1].content


def data_analysis(state):
    """
    Нода анализа данных

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    question = StrOutputParser().invoke(state["messages"][-1])
    directory_path = state["current_files"]
    # Retrieval
    dfs = collect_dfs(directory_path=directory_path)
    if len(dfs) == 0:
        song = StrOutputParser().invoke(answer_chain.invoke(question)["messages"][-1])
    else:
        agentus = create_pandas_dataframe_agent(llm, dfs, "gigachat-functions")
        song = StrOutputParser().invoke(agentus.invoke(state))
    return {"messages": [HumanMessage(content=song["output"], name="Data Analyst")]}


repl = PythonREPL()


@tool
def python_repl(
    code: str,
):
    """
    Если у вас есть код на python применяйте данный инструмент.
    Используйте этот инструмент для выполнения кода на python. Если вам нужно вывести результат
    используйте функцию `print(...)`, она выведет вам результат.

    Args:
        code (str): Код для исполнения

    Returns:
        str: Результат исполнения кода
    """
    if "```" in code:
        code = code.replace("python", "")
        code = code.split("```")[0]
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Succesfully executed:\n```python\n{code}\n```\nStdout: {result}"


python_executing_agent = create_agent(
    llm,
    [python_repl, research_tool],
    "Ты эксперт в программировании и твоя задача решить задачу для научного исследования с использованием инструмента python_repl который может исполнять код на python",
)
python_executing_node = functools.partial(
    agent_node, agent=python_executing_agent, name="Python Executor"
)

action_supervisor_agent = create_team_supervisor(
    llm,
    "Ты - ассистент научного исследователя который должен принять вопрос/сообщение/проблему/задачу пользователя и распределить её решение "
    " между следующими работниками: Python Executor, Problem Manager, Data Analyst. Учитывая полученный запрос"
    " ответь названием работника который должен действовать следующим. Каждый работник"
    " выполняет свою задачу и ответает на запрос полученным результатом и статусом."
    " Problem Manager - работник который берет изначальную проблему пользователя и проводит исследование для её решения. После его ответа можно отдавать результат сразу пользователю, сказав FINISH"
    " Python Executor - работник который отвечает за выполнение задач на программирование и умеет исполнять код на Python."
    " Data Analyst - работник который отвечает за анализ данных в датасетах загруженных пользователем. Если пользователь задал вопрос к какому то `своему` файлу, вызови его."
    " Если результат по твоему мнению уже получен ,ответь словом FINISH."
    " Учитывая приведенное ниже обсуждение, кто должен действовать следующим?"
    " Выбери одно из: {options}"
    " Или напиши FI",
    ["Python Executor", "Problem Manager", "Data Analyst"],
)

action_graph = StateGraph(ActionTeamState)
action_graph.add_node("Python Executor", python_executing_node)
action_graph.add_node("Problem Manager", problem_node)
action_graph.add_node("Data Analyst", data_analysis)
action_graph.add_node("supervisor", action_supervisor_agent)

# Define the control flow
action_graph.add_edge("Python Executor", "supervisor")
action_graph.add_edge("Problem Manager", "supervisor")
action_graph.add_edge("Data Analyst", "supervisor")
action_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "Python Executor": "Python Executor",
        "Problem Manager": "Problem Manager",
        "Data Analyst": "Data Analyst",
        "FINISH": END,
    },
)


action_graph.set_entry_point("supervisor")
chain = action_graph.compile()


# The following functions interoperate between the top level graph state
# and the state of the research sub-graph
# this makes it so that the states of each graph don't get intermixed
def enter_chain(message: str):
    results = {
        "messages": [HumanMessage(content=message)],
    }
    return results


action_chain = enter_chain | chain


super_supervisor_agent = create_team_supervisor(
    llm,
    "Ты - главный супервизор от которого требуется следить за выполнением сложного запроса. "
    " Ты управляешь следующими отделами: Conversate, Perform Action. Учитывая полученный запрос"
    " ответь названием отдела который должен действовать следующим. Каждый работник"
    " выполняет свою задачу и ответает на запрос полученным результатом и статусом."
    " Conversate - отдел который отвечает за общение с пользователем, к нему можно направить если пользователь задает простой вопрос или делает какое то утверждение"
    " Perform Action - отдел который выполняет действия, такие как редактирование документов, исполнение кода, анализа предметной области и генерации гипотез."
    " Если запрос уже является завершенным, ответь словом FINISH."
    " Учитывая приведенное ниже обсуждение, кто должен действовать следующим?"
    " Или же мы должны FINISH? Выбери одно из: {options}",
    ["Conversate", "Perform Action"],
)


class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str
    current_files: str


def get_last_message(state: State) -> str:
    return state["messages"][-1].content


def join_graph(response: dict):
    return {"messages": [response["messages"][-1]]}


super_graph = StateGraph(State)
# First add the nodes, which will do the work
super_graph.add_node("Conversate", get_last_message | answer_chain | join_graph)
super_graph.add_node("Perform Action", get_last_message | action_chain | join_graph)
super_graph.add_node("supervisor", super_supervisor_agent)

# Define the graph connections, which controls how the logic
# propagates through the program
super_graph.add_edge("Conversate", "supervisor")
super_graph.add_edge("Perform Action", "supervisor")
super_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "Conversate": "Conversate",
        "Perform Action": "Perform Action",
        "FINISH": END,
    },
)
super_graph.set_entry_point("supervisor")
super_graph = super_graph.compile()

# {
#        "messages": [
#            HumanMessage(
#                content="Какая сейчас погода"
#            )
#        ],
#    },
#    {"recursion_limit": 150},


def show_help(message: telebot.types.Message) -> NoReturn:
    help_text = """
    Приветствуем вас в нашем мультиагентном ассистенте для научных исследований.
    Данный чат бот позволяет выстроить процесс научного исследования от постановки задачи
    До проведения экспериментов(анализ данных, программы на python)
    Доступные команды:
    /help - Показать список доступных команд
    /files - Показать все файлы, которые вы отправили
    /redact - Форматировать ваш docx файл
    """
    bot.send_message(message.chat.id, help_text)


def get_main_menu_keyboard() -> telebot.types.ReplyKeyboardMarkup:
    keyboard = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    buttons = [
        telebot.types.KeyboardButton("/help"),
        telebot.types.KeyboardButton("/files"),
        telebot.types.KeyboardButton("/redact"),
    ]
    keyboard.add(*buttons)
    return keyboard


@bot.message_handler(commands=["start"])
def start_command(message: telebot.types.Message) -> NoReturn:
    # Create a personal folder for the user
    user_folder = f"files/{message.chat.username}"
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    keyboard = get_main_menu_keyboard()
    bot.send_message(
        message.chat.id,
        "Добро пожаловать! Выберите команду из меню или отправьте сообщение:",
        reply_markup=keyboard,
    )


@bot.message_handler(commands=["help"])
def show_commands(message: telebot.types.Message) -> NoReturn:
    show_help(message)


@bot.message_handler(commands=["files"])
def show_user_files(message: telebot.types.Message) -> NoReturn:
    files = get_user_files(message.chat.id)
    if not files:
        bot.send_message(message.chat.id, "Вы еще не отправляли файлы.")
    else:
        response = "Ваши файлы:\n\n"
        for file_type, file_id, timestamp in files:
            response += f"Тип: {file_type}\nID: {file_id}\nВремя: {timestamp}\n\n"
        bot.send_message(message.chat.id, response)


@bot.message_handler(content_types=["text"])
def echo_message(message: telebot.types.Message) -> NoReturn:
    bot.send_message(
        message.chat.id,
        super_graph.invoke(
            {
                "messages": [HumanMessage(content=message.text)],
                "current_files": f"files/{message.chat.username}/",
            },
            {"recursion_limit": 150},
        ),
    )


@bot.message_handler(content_types=["document", "photo", "audio", "video"])
def handle_files(message: telebot.types.Message) -> NoReturn:
    file_type = message.content_type
    if file_type == "document":
        file_id = message.document.file_id
    elif file_type == "photo":
        file_id = message.photo[-1].file_id
    elif file_type == "audio":
        file_id = message.audio.file_id
    elif file_type == "video":
        file_id = message.video.file_id
    else:
        return

    save_file_info(message.chat.id, message.chat.username, file_type, file_id)
    bot.send_message(message.chat.id, "Это файл")


def save_file_info(user_id: int, username: str, file_type: str, file_id: str) -> None:
    file_info = bot.get_file(file_id)
    file_path = file_info.file_path

    downloaded_file = bot.download_file(file_path)

    file_extension = os.path.splitext(file_path)[1]
    file_name = f"{file_id}{file_extension}"
    file_save_path = f"files/{username}/{file_name}"

    with open(file_save_path, "wb") as new_file:
        new_file.write(downloaded_file)

    save_file_info_to_database(user_id, file_type, file_name)


def save_file_info_to_database(user_id: int, file_type: str, file_name: str) -> None:
    pass


@bot.message_handler(
    content_types=["document", "photo", "audio", "video", "animation", "sticker"]
)
def handle_files(message: telebot.types.Message) -> NoReturn:
    file_type = message.content_type
    if file_type == "document":
        file_id = message.document.file_id
    elif file_type == "photo":
        file_id = message.photo[-1].file_id
    elif file_type == "audio":
        file_id = message.audio.file_id
    elif file_type == "video":
        file_id = message.video.file_id
    elif file_type == "animation":
        file_id = message.animation.file_id
    elif file_type == "sticker":
        bot.send_message(message.chat.id, "Классный стикер!")
        return
    else:
        return

    save_file_info(message.chat.id, message.chat.username, file_type, file_id)
    bot.send_message(message.chat.id, "Это файл")


def save_file_info(user_id: int, username: str, file_type: str, file_id: str) -> None:
    # Получение информации о файле
    file_info = bot.get_file(file_id)
    file_path = file_info.file_path

    # Скачивание файла
    downloaded_file = bot.download_file(file_path)

    # Определение пути для сохранения файла на сервере
    file_extension = os.path.splitext(file_path)[1]
    file_name = f"{file_id}{file_extension}"
    file_save_path = f"files/{username}/{file_name}"

    # Сохранение файла на сервере
    with open(file_save_path, "wb") as new_file:
        new_file.write(downloaded_file)

    # Сохранение информации о файле в базе данных
    save_file_info_to_database(user_id, file_type, file_name)


@bot.message_handler(commands=["files"])
def show_user_files(message: telebot.types.Message) -> NoReturn:
    user_folder = f"files/{message.chat.username}"
    if not os.path.exists(user_folder):
        bot.send_message(message.chat.id, "Вы еще не отправляли файлы.")
        return

    files = os.listdir(user_folder)
    if not files:
        bot.send_message(message.chat.id, "Вы еще не отправляли файлы.")
    else:
        for file_name in files:
            file_path = os.path.join(user_folder, file_name)
            if file_name.endswith(".jpg") or file_name.endswith(".png"):
                with open(file_path, "rb") as photo_file:
                    bot.send_photo(message.chat.id, photo_file)
            elif file_name.endswith(".mp3") or file_name.endswith(".wav"):
                with open(file_path, "rb") as audio_file:
                    bot.send_audio(message.chat.id, audio_file)
            elif file_name.endswith(".mp4") or file_name.endswith(".avi"):
                with open(file_path, "rb") as video_file:
                    bot.send_video(message.chat.id, video_file)
            elif file_name.endswith(".gif"):
                with open(file_path, "rb") as gif_file:
                    bot.send_animation(message.chat.id, gif_file)
            else:
                with open(file_path, "rb") as document_file:
                    bot.send_document(message.chat.id, document_file)


if __name__ == "__main__":
    bot.infinity_polling()
