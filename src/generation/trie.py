from llm_sdk import Small_LLM_Model
from src.models import FunctionFormat


class TrieNode:
    def __init__(self) -> None:
        self.children: dict[int, "TrieNode"] = {}
        self.is_end: bool = False
        self.function_name: str | None = None


class FunctionNameTrie:
    def __init__(
        self, model: Small_LLM_Model, functions: list[FunctionFormat]
    ) -> None:
        self.model = model
        self.root = TrieNode()
        self._build_trie(functions)

    def _build_trie(self, functions: list[FunctionFormat]) -> None:
        for func in functions:
            current_node = self.root
            token_ids: list[int] = self.model.encode(func.name).tolist()[0]
            for token_id in token_ids:
                if token_id not in current_node.children:
                    current_node.children[token_id] = TrieNode()
                current_node = current_node.children[token_id]
            current_node.is_end = True
            current_node.function_name = func.name

    def get_allowed_tokens(self, current_node: TrieNode) -> list[int]:
        return list(current_node.children.keys())
