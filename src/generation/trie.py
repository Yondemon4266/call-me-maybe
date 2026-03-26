"""Trie utilities to constrain generated function names by tokens."""

from llm_sdk import Small_LLM_Model
from src.models import FunctionFormat


class TrieNode:
    """Represent a node in the token trie for function names."""

    def __init__(self) -> None:
        """Initialize an empty trie node.

        Returns:
            None.
        """
        self.children: dict[int, "TrieNode"] = {}
        self.is_end: bool = False
        self.function_name: str | None = None


class FunctionNameTrie:
    """Store valid function names as token paths in a trie."""

    def __init__(
        self, model: Small_LLM_Model, functions: list[FunctionFormat]
    ) -> None:
        """Build a trie from function names encoded by the model tokenizer.

        Args:
            model: Language model wrapper providing tokenization.
            functions: Functions whose names define valid trie paths.

        Returns:
            None.
        """
        self.model = model
        self.root = TrieNode()
        self._build_trie(functions)

    def _build_trie(self, functions: list[FunctionFormat]) -> None:
        """Insert each function name token sequence into the trie.

        Args:
            functions: Function specifications to index.

        Returns:
            None.
        """
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
        """Return valid next token IDs from the provided trie node.

        Args:
            current_node: Node representing current decoding prefix.

        Returns:
            Child token IDs that can legally follow.
        """
        return list(current_node.children.keys())
