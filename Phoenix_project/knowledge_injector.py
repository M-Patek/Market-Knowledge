from ai.prompt_manager import PromptManager
from ai.source_credibility import SourceCredibilityStore
from data_manager import DataManager
from registry import registry
from observability import get_logger

# Configure logger for this module (Layer 12)
logger = get_logger(__name__)


def inject_rules(data_manager: DataManager, rules: dict):
    """Injects symbolic rules into the prompt and credibility systems."""
    logger.info(f"Injecting {len(rules.get('symbolic_rules', []))} symbolic rules...")
    prompt_manager: PromptManager = registry.resolve("prompt_manager")
    credibility_store: SourceCredibilityStore = registry.resolve("credibility_store")

    for rule in rules.get('symbolic_rules', []):
        if rule['action'] == 'update_prompt':
            prompt_manager.update_system_prompt_template(
                rule['key'], rule['template']
            )
        elif rule['action'] == 'update_credibility':
            if 'source_type' in rule and 'source_name' in rule:
                credibility_store.update_source_credibility(
                    rule['source_type'], rule['source_name'], rule['value']
                )
    
    logger.info("Symbolic rules injection complete.")
