from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from state import RelevanceState

from node import(
    # extract_product_features,
    extract_patent_features,
    final_analysis
)

# Graph
graph = StateGraph(RelevanceState)

# Nodes
# graph.add_node(
#     "Extract Product Features",
#      extract_product_features
# )

graph.add_node(
    "Extract Patent Features",
     extract_patent_features
)

graph.add_node(
    "Final Analysis",
     final_analysis
)

graph.add_edge(
    START,
    "Extract Patent Features"
    # "Extract Product Features"
)

# graph.add_edge(
#     # "Extract Product Features",
#     "Extract Patent Features"
# )

graph.add_edge(
    "Extract Patent Features",
    "Final Analysis"
)

graph.add_edge(
    "Final Analysis",
    END
)

memory = InMemorySaver()

#Define the workflow
workflow = graph.compile(
    # checkpointer=memory
)

png = workflow.get_graph().draw_mermaid_png()

with open("workflow_3_nodes.png", "wb") as f:
    f.write(png)

print("Workflow image saved as workflow_3_nodes.png")



