from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from tools.llm import LLM


class FieldInformation(BaseModel):
    """Information in the context"""
    field: str = Field(description="the Single value of the field in the context,should be string")


def extraction_keys_info_agent(query):
    parser = PydanticOutputParser(pydantic_object=FieldInformation)

    template = """
    You are the TradeFinance expert.Your task is to extract the value of the given field from the provided context.
    
    Examples:
    Context: "The expiry date of the letter of credit is 15-06-2023."
    Field to look up: expiry date
    Output: 15-06-2023

    Context: "The country of origin mentioned on the commercial invoice is United States."
    Field to look up: country of origin
    Output: United States
    
    
    Field to look up: 
    {field}

    Provide your response in the following format:
    
    \n{format_instructions}
    
    The context for {document} document ::
    {query}
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["query","field","document"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    model = LLM.call_model

    chain = prompt |model | JsonOutputParser()

    res = chain.invoke(query)
    return res


if __name__ == '__main__':
    query = {"query": "the date: 22-10-1999 \n the date: 22-10-1999 \n the date: 22-10-1999","field":"the date","document":"bill of lading"}
    print(extraction_keys_info_agent(query))
