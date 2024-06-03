# # Define your desired data structure.
# from typing import List
#
# from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
# from langchain_core.prompts import PromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field
# from tools.llm import LLM
#
#
# class Question(BaseModel):
#     """Question for the statement"""
#     question: str = Field(description="Generate variation or reformulated of question for the statement")
#
# class Questions(BaseModel):
#     """five questions for the statement"""
#     questions: List[Question]
#
#
# def generate_questions_agent(query):
#     parser = PydanticOutputParser(pydantic_object=Questions)
#     template = """
#     You have to generate question for the statement
#
#     output format:
#     \n{format_instructions}
#
#     Rule:
#     \n{query}\n"""
#     prompt = PromptTemplate(
#         template=template,
#         input_variables=["query"],
#         partial_variables={"format_instructions": parser.get_format_instructions()},
#     )
#
#     model = LLM.call_model
#
#     chain = prompt | model | JsonOutputParser()
#
#     res = chain.invoke({"query": query})
#     return res
#
#
# if __name__ == '__main__':
#     print(generate_questions_agent("covering schedule date"))


# Define your desired data structure.
from typing import List

from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from tools.llm import LLM


class Variation(BaseModel):
    """variation of the input statement"""
    variation: str = Field(description="A rephrased or reformulated version of the input statement")

class Variations(BaseModel):
    """Multiple variations of the input statement"""
    variations: List[Variation]


def generate_questions_agent(query):
    parser = PydanticOutputParser(pydantic_object=Variations)
    template = """
    Your task is to generate five variations  or rephrasing of the given statement.
    these variations should capture the essence of the original statement but be expressed in different ways to improve the chance of finding relevant information in a data corpus
    
    Input: {query}
    
    Provide your response in the following format:
    
    {format_instructions}
    
    """
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
    print(generate_questions_agent("Issue Date of Bill of Lading"))
