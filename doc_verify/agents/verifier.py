# Define your desired data structure.
from typing import List

from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from tools.llm import LLM


class Verification(BaseModel):
    """Document verification"""
    status: str = Field(description="should be yes/no")
    reason: str = Field(description="reason for the status")

def verifier_agent(input_data):
    parser = PydanticOutputParser(pydantic_object=Verification)
    template = """
    You are the Document Verifier agent
    Given documents and its contexts perform action or operation
    
    First Document Context: {first_document_context}
    First Document: {first_document}
    
    Second Document Context: {second_document_context}
    Second Document: {second_document}
    
    Action to Perform: {action_or_perform}
    output format:
    \n{format_instructions}
    \n"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    model = LLM.call_model

    chain = prompt | model | JsonOutputParser()

    res = chain.invoke(input_data)
    return res



if __name__ == '__main__':
    input_data = {
        "first_document_context": "Context of the first document",
        "first_document": "First document content",
        "second_document_context": "Context of the second document",
        "second_document": "Second document content",
        "action_or_perform": "Verify if the schedule date is covered",
    }
    print(verifier_agent(input_data))
