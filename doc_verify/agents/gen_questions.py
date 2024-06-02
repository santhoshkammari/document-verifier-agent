# Define your desired data structure.
from typing import List

from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from tools.llm import LLM


class Question(BaseModel):
    """Question for the statement"""
    question: str = Field(description="Generate variation or reformulated of question for the statement")

class Questions(BaseModel):
    """five questions for the statement"""
    questions: List[Question]


def generate_questions_agent(query):
    parser = PydanticOutputParser(pydantic_object=Questions)
    template = """
    You have to generate question for the statement

    output format:
    \n{format_instructions}

    Rule:
    \n{query}\n"""
    prompt = PromptTemplate(
        template=template,
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    model = LLM.call_model

    chain = prompt | model | JsonOutputParser()

    res = chain.invoke({"query": query})
    return res


if __name__ == '__main__':
    print(generate_questions_agent("covering schedule date"))
