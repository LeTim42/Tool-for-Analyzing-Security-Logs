import json
import uuid
import ollama
from langchain_ollama import OllamaLLM
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph


class LLM:
    def __init__(self, model, context_size):
        self.context_size = context_size
        model = OllamaLLM(
            model=model,
            callbacks=CallbackManager([StreamingStdOutCallbackHandler()]),
            temperature=0.0,
            num_ctx=context_size
        )

        system_prompt = """
        You are a cybersecurity expert. Analyze logs and identify suspicious events.

        Return a report in JSON format as a list of anomalies with keys:
        "type": "Type of anomaly",
        "datetime": "Jan 01 13:37:42",
        "file": "Path to file",
        "description": "Description of anomaly",
        "severity": "Low / Medium / High"

        List all anomalies separated by commas, enclosing the entire list in square brackets.
        Your message must use ONLY JSON format! No extra text outside of JSON!
        """
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        def call_model(state: MessagesState):
            prompt = prompt_template.invoke(state)
            try:
                response = model.invoke(prompt)
            except ollama.ResponseError as e:
                print(f"Error: {e.error}. Run command 'ollama pull' to download model")
                exit(1)
            return {"messages": response}

        workflow = StateGraph(state_schema=MessagesState)
        workflow.add_edge(START, "model")
        workflow.add_node("model", call_model)

        memory = MemorySaver()
        self.llm = workflow.compile(checkpointer=memory)

    def analyze_logs(self, logs):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.context_size, separators=["\n"])
        chunks = text_splitter.split_text(logs)
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        report = []
        for chunk in chunks:
            output = self.llm.invoke({"messages": [HumanMessage(chunk)]}, config)
            last_output = output["messages"][-1].content
            if '[' in last_output and ']' in last_output:
                last_output = f"[{last_output.split('[', 1)[-1].rsplit(']', 1)[0]}]"
            else:
                last_output = f"[{last_output}]"
            report.extend(json.loads(last_output))
        return report
