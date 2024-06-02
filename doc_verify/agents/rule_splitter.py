# Define your desired data structure.
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel,Field
from tools.llm import LLM

class RuleSplitter(BaseModel):
    """Separate the Rule into two different statements of source and destination"""
    source_statement: str = Field(description="source document statement mentioned")
    destination_statement: str = Field(description="destination document statement mentioned")
    operation: str = Field(description="operation or action to perform")

def rule_splitter_agent(query):
    parser = PydanticOutputParser(pydantic_object=RuleSplitter)
    template = """
    You are the Rulesplitter Expert.
    
    Instructions:
    1. Generally the text is information about source document and its keys , destination document and its keys,the operation to perform between them 
     these three things can be found in any order.
     
    Given a rule you need to split it into two parts, each part should have its own single document and information.
    
    Provide the response in the following format:
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
    print(rule_splitter_agent("the bill of lading document date should be smaller than covering schedule date"))
