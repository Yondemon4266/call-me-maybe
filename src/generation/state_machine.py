from collections import deque

from llm_sdk import Small_LLM_Model
from src.generation.decoding_steps import DecodingStep
from src.generation.trie import FunctionNameTrie
from src.models import FunctionFormat


class JsonStateMachine:
    def __init__(
        self,
        model: Small_LLM_Model,
        prompt_text: str,
        functions: list[FunctionFormat],
        fn_name_trie: FunctionNameTrie,
    ):
        self.model = model
        self.functions = functions
        self.fn_name_trie = fn_name_trie

        # 1. Définition des textes statiques
        self.static_steps_text = {
            DecodingStep.STEP_START_TO_PROMPT: '{\n  "prompt": "',
            DecodingStep.STEP_TO_NAME: '",\n  "name": "',
            DecodingStep.STEP_TO_PARAMS: '",\n  "parameters": {',
            DecodingStep.STEP_END: "\n}",
        }

        # 2. Encodage des étapes statiques
        self.encoded_static_steps: dict[DecodingStep, list[int]] = {
            step: self.model.encode(text).tolist()[0]
            for step, text in self.static_steps_text.items()
        }
        # On ajoute le prompt utilisateur dynamiquement
        self.encoded_static_steps[DecodingStep.STEP_PROMPT_VALUE] = (
            self.model.encode(prompt_text).tolist()[0]
        )

        # 3. Transitions FSM
        self.transitions = {
            DecodingStep.STEP_START_TO_PROMPT: DecodingStep.STEP_PROMPT_VALUE,
            DecodingStep.STEP_PROMPT_VALUE: DecodingStep.STEP_TO_NAME,
            DecodingStep.STEP_TO_NAME: DecodingStep.AI_NAME_VALUE,
            DecodingStep.AI_NAME_VALUE: DecodingStep.STEP_TO_PARAMS,
            DecodingStep.STEP_TO_PARAMS: DecodingStep.AI_PARAMS_KEYS,
            DecodingStep.AI_PARAMS_KEYS: DecodingStep.AI_PARAMS_VALS,
            DecodingStep.AI_PARAMS_VALS: DecodingStep.STEP_END,
        }

        # 4. Variables d'état (La mémoire de l'arbitre)
        self.current_step = DecodingStep.STEP_START_TO_PROMPT
        self.current_step_queue = deque(
            self.encoded_static_steps[self.current_step]
        )
        self.current_trie_node = self.fn_name_trie.root

        # Variables pour les paramètres de la fonction choisie
        self.chosen_function = None
        self.current_params_queue = []
        self.current_param_type = None
        self.is_first_param = True

    def get_allowed_tokens(self) -> list[int]:
        """Retourne la liste des Token IDs autorisés pour l'état actuel."""
        match self.current_step:
            case (
                DecodingStep.STEP_START_TO_PROMPT
                | DecodingStep.STEP_PROMPT_VALUE
                | DecodingStep.STEP_TO_NAME
                | DecodingStep.STEP_TO_PARAMS
                | DecodingStep.STEP_END
            ):
                return [self.current_step_queue[0]]

            case DecodingStep.AI_NAME_VALUE:
                return self.fn_name_trie.get_allowed_tokens(
                    self.current_trie_node
                )

            case DecodingStep.AI_PARAMS_KEYS:
                # Bientôt : Forcer les clés des paramètres
                return []

            case DecodingStep.AI_PARAMS_VALS:
                # Bientôt : Filtrer le vocabulaire (Regex)
                return []

        return []

    def advance(self, token_id: int) -> bool:
        """
        Fait avancer la machine à états avec le token choisi par l'IA.
        Retourne True si la génération JSON est complètement terminée, False sinon.
        """
        match self.current_step:
            case (
                DecodingStep.STEP_START_TO_PROMPT
                | DecodingStep.STEP_PROMPT_VALUE
                | DecodingStep.STEP_TO_NAME
                | DecodingStep.STEP_TO_PARAMS
            ):
                self.current_step_queue.popleft()
                if not self.current_step_queue:
                    self._move_to_next_step()

            case DecodingStep.STEP_END:
                self.current_step_queue.popleft()
                if not self.current_step_queue:
                    return True  # C'est la fin absolue du JSON !

            case DecodingStep.AI_NAME_VALUE:
                self.current_trie_node = self.current_trie_node.children[
                    token_id
                ]
                if self.current_trie_node.is_end:
                    self.chosen_function = self.current_trie_node.function_name
                    self._load_function_parameters()
                    self._move_to_next_step()
                    self.current_trie_node = self.fn_name_trie.root

            case DecodingStep.AI_PARAMS_KEYS:
                pass  # Bientôt : gérer l'avancée de la clé dynamique

            case DecodingStep.AI_PARAMS_VALS:
                pass  # Bientôt : gérer l'avancée de la valeur

        return False

    def _move_to_next_step(self):
        """Méthode utilitaire interne pour passer à l'étape suivante."""
        self.current_step = self.transitions[self.current_step]
        if self.current_step in self.encoded_static_steps:
            self.current_step_queue = deque(
                self.encoded_static_steps[self.current_step]
            )

    def _load_function_parameters(self):
        """Méthode utilitaire interne pour charger les paramètres quand la fonction est trouvée."""
        func_spec = next(
            f for f in self.functions if f.name == self.chosen_function
        )
        self.current_params_queue = list(func_spec.parameters.items())
        self.is_first_param = True

    # --- MÉTHODE POUR LE RACCOURCI FAST-FORWARD ---
    def fast_forward_prompt(self) -> list[int]:
        """Récupère les tokens du prompt et passe directement à l'état suivant."""
        prompt_tokens = self.encoded_static_steps[
            DecodingStep.STEP_PROMPT_VALUE
        ]
        self._move_to_next_step()
        return prompt_tokens
