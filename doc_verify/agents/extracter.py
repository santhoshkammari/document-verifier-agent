from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from tools.llm import LLM

class Information(BaseModel):
    """Information in the document"""
    document: str = Field(description="the document or context being referred to", default=None)
    field: str = Field(description="the specific field, concept, or aspect being discussed", default=None)

def extraction_agent(statement,query):
    parser = PydanticOutputParser(pydantic_object=Information)
    template = """
    Given an input text related to trade finance import/export rules or contracts, your task is to identify the following first check in Statement if not found you can use DATA:

    1. The document or context being referred to (e.g., trade finance import/export rules, contract, etc.).
    2. The specific field, concept, or aspect being discussed in the input text.

    Examples:
    Input: "Statement: The expiry date of the letter of credit  DATA: should not exceed 90 days from the date of shipment."
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
    Statement: {statement} DATA: {query}
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    model = LLM.call_model  # Use a more advanced language model

    chain = prompt | model | parser

    res = chain.invoke({"statement":statement,"query": query})
    try:
        res = {"document":res.document,
           "field":res.field}
    except:
        res = {}

    return res

if __name__ == '__main__':
    input_text = "should be less than the date of covering schedule"
    print(extraction_agent(statement="bill of lading",
                           query="country of origin is it same in commercial invoice and bill of lading  "))