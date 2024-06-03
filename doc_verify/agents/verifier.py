# # Define your desired data structure.
# from typing import List
#
# from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
# from langchain_core.prompts import PromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field
# from doc_verify.tools.llm import LLM
#
#
# class Verification(BaseModel):
#     """Document verification"""
#     status: str = Field(description="should be yes/no")
#     reason: str = Field(description="reason for the status")
#
# def verifier_agent(input_data):
#     parser = PydanticOutputParser(pydantic_object=Verification)
#     template = """
#     You are the Document Verifier agent
#     Given documents and its contexts perform action or operation
#
#     First Document Context: {first_document_context}
#     First Document: {first_document}
#
#     Second Document Context: {second_document_context}
#     Second Document: {second_document}
#
#     Action to Perform: {action_or_perform}
#     output format:
#     \n{format_instructions}
#     \n"""
#
#     prompt = PromptTemplate(
#         template=template,
#         input_variables=["query"],
#         partial_variables={"format_instructions": parser.get_format_instructions()}
#     )
#
#     model = LLM.call_model
#
#     chain = prompt | model | JsonOutputParser()
#
#     res = chain.invoke(input_data)
#     return res
#
#
#
# if __name__ == '__main__':
#     input_data = {
#         "first_document_context": "ZIMUOSS160139",
#         "first_document": "Bill of Lading",
#         "first_document_field":"Number",
#         "second_document_context": "BT/EXP/1472",
#         "second_document": "Commercial Invoice",
#         "second_document_field":"Number",
#         "action_or_perform": "matches",
#     }
#     print(verifier_agent(input_data))


from typing import List, Literal

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from tools.llm import LLM

class Verification(BaseModel):
    """Document verification"""
    status: Literal['yes', 'no'] = Field(description="'yes' if the action is satisfied, 'no' otherwise")
    reason: str = Field(description="detailed explanation for the provided status")

def verifier_agent(input_data):
    parser = PydanticOutputParser(pydantic_object=Verification)
    template = """
    You are the Document Verifier agent. Your task is to verify if the given fields from two documents satisfy the specified action or operation.

    First Document: {first_document}
    Field: {first_document_field}
    Field Value: {first_document_context}

    Second Document: {second_document}
    Field: {second_document_field}
    Field Value: {second_document_context}

    Action or Operation to Perform: {action_or_perform}

    To provide the answer, follow these steps:
    1. Understand the action or operation to be performed on the field values.
    2. Perform the specified action or operation on the field values.
    3. Determine if the action or operation is satisfied or not.
    4. Respond with 'yes' or 'no' based on the result, and provide a reason.

    Your response should be in the following format:
    {format_instructions}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["first_document_context", "first_document", "first_document_field",
                         "second_document_context", "second_document", "second_document_field",
                         "action_or_perform"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    model = LLM.call_model

    chain = prompt | model | parser

    if 'not found' in [input_data["first_document_context"],input_data["second_document_context"]]:
        res = {"status": "not found", "reason": "not found", "state": input_data.get("state", {})}
    else:
        res = chain.invoke(input_data)

    res = {"status":res.status,
           "reason":res.reason,
           "state":input_data.get("state",{})}
    return res

if __name__ == '__main__':
    input_data = {
        "first_document_context": "ZIMUOSS160139",
        "first_document": "Bill of Lading",
        "first_document_field": "Number",
        "second_document_context": "BT/EXP/1472",
        "second_document": "Commercial Invoice",
        "second_document_field": "Number",
        "action_or_perform": "matches",
        "state":{"state":"adsfsdf"}
    }
    print(verifier_agent(input_data))