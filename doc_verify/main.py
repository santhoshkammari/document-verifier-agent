import inspect
import io
import operator
import time
from typing import Annotated

from PIL import Image
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from operations import Operation


def reducer(existing: dict, new: dict) -> dict:
    return {**existing, **new}



def debug(func,*args, **kwargs):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        print('*'*50)
        print(f"Calling {func.__name__}")
        result =  func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Agent {func.__name__} took {end - start:.4f} seconds")
        print('='*50)

        return result
    return wrapper

def get_class_methods(cls):
    names = inspect.getmembers(cls,predicate=inspect.isfunction)
    return {name for name, _ in names}

def rule_agent(query):
    graph = StateGraph(Annotated[dict,reducer])

    obj = Operation()


    methods = get_class_methods(Operation)

    for method in methods:
        if not method.startswith('__'):
            graph.add_node(method,getattr(obj,method))

    # graph.add_node("split_agent",split_agent)
    # graph.add_node("generate_questions_first",generate_questions_first)
    # graph.add_node("generate_questions_second",generate_questions_second)
    # graph.add_node("retreive_agent_first",retreive_agent_first)
    # graph.add_node("retreive_agent_second",retreive_agent_second)
    # graph.add_node("extract_agent_first",extract_agent_first)
    # graph.add_node("extract_agent_second",extract_agent_second)
    # graph.add_node("verifier_agent",verifier_agent)
    # graph.add_node("keys_extraction_first",keys_extraction_first)
    # graph.add_node("keys_extraction_second",keys_extraction_second)
    # graph.add_node("statement_first",statement_first)
    # graph.add_node("statement_second",statement_second)

    graph.add_edge("split_agent","statement_first")
    graph.add_edge("split_agent","statement_second")

    graph.add_edge("statement_first","keys_extraction_first")
    graph.add_edge("keys_extraction_first","extract_agent_first")
    graph.add_edge("statement_first","generate_questions_first")

    # graph.add_edge("generate_questions_first","retreive_agent_first")
    # graph.add_edge("retreive_agent_first","extract_agent_first")
    graph.add_edge("generate_questions_first","extract_agent_first")

    #
    graph.add_edge("statement_second","generate_questions_second")

    # graph.add_edge("generate_questions_second","retreive_agent_second")
    # graph.add_edge("retreive_agent_second","extract_agent_second")
    graph.add_edge("generate_questions_second","extract_agent_second")
    graph.add_edge("statement_second","keys_extraction_second")
    graph.add_edge("keys_extraction_second","extract_agent_second")

    graph.add_edge("extract_agent_first","verifier_agent")
    graph.add_edge("extract_agent_second","verifier_agent")

    graph.set_entry_point("split_agent")
    graph.set_finish_point("verifier_agent")

    app  = graph.compile()

    res = app.invoke({"query":query})
    return res
    # img_name = "graph.png"
    # Image.open(io.BytesIO(app.get_graph().draw_mermaid_png())).save(img_name)

if __name__ == '__main__':
    rule_agent("the bill of lading date shoulde be less thatn the date of covering schedule")