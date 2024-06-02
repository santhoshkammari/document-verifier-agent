import inspect
import json
import os
import time
from copy import deepcopy

from agents.extract_keys_info_agent import extraction_keys_info_agent
from agents.extracter import extraction_agent
from agents.gen_questions import generate_questions_agent
from agents.rule_splitter import rule_splitter_agent
from agents.verifier import verifier_agent
from tools.doc_similarities_retreiver import retreive_document_information


def save_data(data,name):
    os.makedirs("debug", exist_ok=True)
    new_data = deepcopy(data)
    with open(f"debug/{name}.json", "w") as f:
        json.dump(new_data, f, indent=2)
def debug(func,*args, **kwargs):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        print(f" ========== Loading {func.__name__} ==========")
        result =  func(*args, **kwargs)
        end = time.perf_counter()
        print(f"********** Agent {func.__name__} took {end - start:.4f} seconds **********")
        save_data(result,func.__name__)
        return result
    return wrapper


def retreive_agent_first(state: dict):
    res = retreive_document_information(document=state["first_statement_extraction"]["document"],
                                        questions=state["questions_first_statement"]["questions"])
    return res

def retreive_agent_second(state: dict):
    res = retreive_document_information(document=state["second_statement_extraction"]["document"],
                                        questions=state["questions_second_statement"]["questions"])
    return res
class Operation:

    def __getattribute__(self, item):
        attr = object.__getattribute__(self, item)
        if callable(attr) and not item.startswith("__"):
            return debug(attr)
        return attr


    def split_agent(self,state: dict):
        res = rule_splitter_agent(state["query"])
        # res = {
        #   "query": "the bill of lading date shoulde be less thatn the date of covering schedule",
        #   "source_statement": "the bill of lading date",
        #   "destination_statement": "should be less than the date of covering schedule",
        #   "operation": "less than"
        # }
        state["source_statement"] = res.get("source_statement")
        state["destination_statement"] = res.get("destination_statement")
        state["operation"] = res.get("operation")
        return state


    def generate_questions_first(self,state: dict):
        res = generate_questions_agent(state["first_statement"])
        # res = {
        #     "questions": [
        #       {
        #         "question": "What is the significance of the bill of lading date?"
        #       },
        #       {
        #         "question": "How does the bill of lading date impact the shipping process?"
        #       },
        #       {
        #         "question": "Can you provide an example of a situation where the bill of lading date is crucial?"
        #       },
        #       {
        #         "question": "What are the potential consequences if the bill of lading date is incorrect?"
        #       },
        #       {
        #         "question": "How does the bill of lading date relate to other important shipping documents?"
        #       }
        #     ]
        #   }
        state["questions_first_statement"] = res

        return state

    def generate_questions_second(self,state: dict):
        res = generate_questions_agent(state["second_statement"])
        # res = {"questions": ["sample4", "sample5", "sample6"]}
        state["questions_second_statement"] = res

        return state




    def keys_extraction_first(self,state: dict):
        res = extraction_agent(state["source_statement"])
        # res = {
        #     "document": "Bill of Lading",
        #     "field": "Date",
        #     "key": "2023-02-15"
        #   }
        state["first_statement_extraction"] = res
        return state

    def keys_extraction_second(self,state: dict):
        res = extraction_agent(state["destination_statement"])
        # res = {"document": "second doc title",
        #        "key": "keyname",
        #        "field": "fieldname"}
        state["second_statement_extraction"] = res
        return state


    def extract_agent_first(self,state: dict):
        state["first_state_embeds"] = retreive_agent_first(state)
        query = {"query": "\n".join(state["first_state_embeds"]),
                 "field": state.get("first_statement_extraction", {}).get("field"),
                 "document": state.get("first_statement_extraction", {}).get("document")}
        res = extraction_keys_info_agent(query)
        # res = {'field': ['22-10-1999', '22-10-1999', '22-10-1999']}
        res = res.get("field","Not Found")
        state["first_document_context"] = " or ".join(res) if isinstance(res,list) else res
        return state

    def extract_agent_second(self,state: dict):
        state["second_state_embeds"] = retreive_agent_second(state)
        query = {"query": "\n".join(state["second_state_embeds"]),
                 "field": state.get("second_statement_extraction", {}).get("field"),
                 "document": state.get("second_statement_extraction", {}).get("document")}
        res = extraction_keys_info_agent(query)
        # res = {'field': ['22-10-1999', '22-10-1999', '22-10-1999']}
        res = res.get("field", "Not Found")
        state["second_document_context"] = " or ".join(res) if isinstance(res, list) else res
        return state

    def verifier_agent(self,state: dict):
        input_data = {
            "first_document_context": state["first_document_context"] + state[
                "first_document_context"],
            "first_document": state["first_statement_extraction"].get("document"),
            "second_document_context": state["second_document_context"] + state[
                "second_document_context"],
            "second_document": state["second_statement_extraction"].get("document"),
            "action_or_perform": state["operation"]
        }
        res = verifier_agent(input_data)
        # res = {}
        state["status"] = res.get("status")
        state["reason"] = res.get("reason")
        return state


    def statement_first(self,state: dict):
        state["first_statement"] = state["source_statement"]
        return state


    def statement_second(self,state: dict):
        state["second_statement"] = state["destination_statement"]
        return state


if __name__ == '__main__':
    obj = Operation()
    # obj= Operation()
    # print(obj.split_agent({"query":"the date in bill of lading should be less thatn the date in covering schedule"}))