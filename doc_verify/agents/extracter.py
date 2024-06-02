from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from tools.llm import LLM

class Information(BaseModel):
    """Information in the document"""
    document: str = Field(description="the document or context being referred to", default=None)
    field: str = Field(description="the specific field, concept, or aspect being discussed", default=None)

def extraction_agent(query):
    parser = PydanticOutputParser(pydantic_object=Information)
    template = """
    Given an input text related to trade finance import/export rules or contracts, your task is to identify the following:

    1. The document or context being referred to (e.g., trade finance import/export rules, contract, etc.).
    2. The specific field, concept, or aspect being discussed in the input text.

    Examples:
    Input: "The expiry date of the letter of credit should not exceed 90 days from the date of shipment."
    Output:
        "document": "Letter of Credit",
        "field": "Expiry date"

    Input: "The country of origin should be clearly stated on the commercial invoice."
    Output:
        "document": "Commercial Invoice",
        "field": "Country of origin"

    Provide your response in the following format:

    {format_instructions}

    Input:
    {query}
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    model = LLM.call_model  # Use a more advanced language model

    chain = prompt | model | parser

    res = chain.invoke({"query": query})
    try:
        res = {"document":res.document,
           "field":res.field}
    except:
        res = {}

    return res

if __name__ == '__main__':
    input_text = "should be less than the date of covering schedule"
    print(extraction_agent(input_text))