import io
import time
from typing import List

from PIL import Image
from langgraph.graph import MessageGraph, END
from langchain_core.messages import HumanMessage


async def timed():
    print("timed")
    time.sleep(1)
    return "hai"

def example():
    print("example")
    return "hai"
async def add_one(input: List[HumanMessage]):
    print("a",input[0].content)
    input[0].content = input[0].content + 'a'
    x =  await timed()
    y = example()
    return input


graph = MessageGraph()

graph.add_node('branch_a',add_one)



graph.set_entry_point('branch_a')
graph.set_finish_point('branch_a')

runnable = graph.compile()

# Image.open(io.BytesIO(runnable.get_graph().draw_mermaid_png())).show()
import asyncio

async def main():
    result = await runnable.ainvoke("p")
    print(result)

# Run the main function using asyncio
asyncio.run(main())

