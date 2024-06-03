# from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
# from langchain_core.prompts import PromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field
# from doc_verify.tools.llm import LLM
#
# class Field(BaseModel):
#     """Information in the context"""
#     field: str = Field(description="the Single value of the field in the context, should be a string")
#
# def extraction_keys_info_agent(query):
#     parser = PydanticOutputParser(pydantic_object=Field)
#
#     template = """
#     You are an expert in extracting specific information from documents. Your task is to extract the value of the given field from the provided context.
#
#
#     To provide the answer, follow these steps:
#     1. Carefully read the context and understand the content.
#     2. Identify the part of the context that corresponds to the requested field.
#     3. Extract the relevant value from that part.
#     4. If the requested field is not present in the context, respond with "Field not found."
#     5. If the context is unclear or ambiguous, respond with "Context unclear."
#
#     Context: "{query}"
#     Field to look up: {field}
#
#     Your response should be in the following format:
#     {format_instructions}
#     """
#     prompt = PromptTemplate(
#         template=template,
#         input_variables=["query", "field"],
#         partial_variables={"format_instructions": parser.get_format_instructions()},
#     )
#
#     model = LLM.call_model
#
#     chain = prompt | model
#
#     res = chain.invoke(query)
#
#     try:
#         res = parser.parse(res)
#     except:
#         try:
#             output_parser = JsonOutputParser()
#             res = output_parser.parse(res)
#         except:
#             pass
#
#
#     return res
import json

from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from doc_verify.tools.llm import LLM

class Field(BaseModel):
    """Information in the context"""
    field: str = Field(description="the Single value of the field in the context, should be a string")

def extraction_keys_info_agent(query):
    parser = PydanticOutputParser(pydantic_object=Field)

    template = """
    You are an expert in extracting specific information from documents. Your task is to extract the value of the given field from the provided context.

    Context: "{query}"
    Document: {document}
    Field to look up: {field}

    To provide the answer, follow these steps:
    1. Carefully read the context and understand the content.
    2. Identify the part of the context that corresponds to the requested field.
    3. Extract the relevant value from that part.
    4. If the requested field is not present in the context, respond with "Field not found."
    5. If the context is unclear or ambiguous, respond with "Context unclear."

    Your response should be in the following format:
    {format_instructions}
    """

    template_v1_0_0 = """
    Extract the value of the given field from the provided context, even if the context is unstructured or disorganized.

    Context: {query}
    Document Type: {document}
    Field: {field}

    Instructions:
    1. Examine the context carefully, which may consist of short phrases, incomplete sentences, or unordered text fragments.
    2. Identify any relevant pieces of information that could correspond to the requested field.
    3. Extract the relevant value(s) from those pieces of information.

    
    Response Format:
    {format_instructions}"""

    template_v1_0_1 ="""
    Extract the value of the specified field from the given context, which may contain unstructured or fragmented text.

    Context: {query}
    Field: {field}

    Instructions:
    1. Scan the context for any text that represents the requested field value.
    2. If found, extract the relevant value(s).
    3. If not found, respond with "Not found".
    4. Separate multiple values with commas.
    5. Preserve any formatting or special characters.

    Examples:
    Context: "John Doe, 25, New York"
    Field: Age
    Output: 25

    Context: "Prices: $10, $15, $20"  
    Field: Prices
    Output: $10, $15, $20

    Context: "Error: File not found at /docs/report.txt"
    Field: Error Message
    Output: File not found at /docs/report.txt

    Response Format:
    {format_instructions}
    """
    prompt = PromptTemplate(
        template=template_v1_0_1,
        input_variables=["query", "field"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    model = LLM.call_model

    chain = prompt | model

    res = chain.invoke(query)

    try:
        res = parser.parse(res)
        res = res.field
    except:
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res)
            res = res["field"]
        except:
            res = "Not found"
    return res

if __name__ == '__main__':
    query = {"query": """PARTICULARS. AS FURNISHED BY SHIPPER\u2019
    "itemized, valusd by. the Marchant prior to
    "GILL OF LADING No,\n\nZIMUOSS160139 ZIMUOSS160139
    "GILL OF LADING No,\n\nZIMUOSS160139 ZIMUOSS160139
    "GILL OF LADING No,\n\nZIMUOSS160139 ZIMUOSS160139
    "itemized, valusd by. the Marchant prior to
    "itemized, valusd by. the Marchant prior to
    "IDESCRIPTION OF GOODS i MEASUREMENT
    "GILL OF LADING No,\n\nZIMUOSS160139 ZIMUOSS160139
    "Goods) and 22 (Law and Jusiscbon}. Tho Package""","field":"Number","document":"bill of lading"}

    temp = json.load(open("doc_verify/debug/extract_agent_first.json"))
    type = "first"
    query["query"] = "\n".join(temp[f"{type}_state_embeds"])
    query["field"] = temp[f"{type}_statement_extraction"]["field"]
    query["document"] = temp[f"{type}_statement_extraction"]["document"]
    print(query)

    print(extraction_keys_info_agent(query))
