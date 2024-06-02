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


from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from tools.llm import LLM

class Field(BaseModel):
    """Information in the context"""
    field: str = Field(description="the Single value of the field in the context, should be a string")

def extraction_keys_info_agent(query):
    parser = PydanticOutputParser(pydantic_object=Field)

    template = """
    You are an expert in extracting specific information from documents. Your task is to extract the value of the given field from the provided context.

    Context: "{query}"
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
    prompt = PromptTemplate(
        template=template,
        input_variables=["query", "field"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    model = LLM.call_model

    chain = prompt | model | parser

    res = chain.invoke(query)

    try:
        res = res.field
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
    print(extraction_keys_info_agent(query))
