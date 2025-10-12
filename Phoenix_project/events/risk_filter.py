# events/risk_filter.py
"""
Implements the first-stage, high-performance synchronous event filter.
Its sole purpose is to quickly discard irrelevant news items based on a
pre-defined keyword logic before they are sent for expensive AI analysis.
"""
import yaml
import re
import logging
from typing import List, Set

class RiskFilter:
    """
    A synchronous, keyword-based filter to identify potentially systemic events.
    """
    def __init__(self, config_path: str = "config/event_filter_config.yaml"):
        """
        Initializes the filter by loading and compiling keyword sets from the config.

        Args:
            config_path: Path to the event_filter_config.yaml file.
        """
        self.logger = logging.getLogger("PhoenixProject.RiskFilter")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Store keyword sets in lowercase for case-insensitive matching
            self._risk_actions: Set[str] = {kw.lower() for kw in config.get('RiskActionKeywords', [])}
            self._systemic_entities: Set[str] = {kw.lower() for kw in config.get('SystemicEntityKeywords', [])}
            self._exclusions: Set[str] = {kw.lower() for kw in config.get('ExclusionKeywords', [])}

            # Pre-compile regex patterns for precise, whole-word matching
            self._risk_actions_regex = self._compile_regex(self._risk_actions)
            self._systemic_entities_regex = self._compile_regex(self._systemic_entities)
            self._exclusions_regex = self._compile_regex(self._exclusions)

            self.logger.info(f"RiskFilter initialized with {len(self._risk_actions)} risk keywords, "
                             f"{len(self._systemic_entities)} entity keywords, and {len(self._exclusions)} exclusion keywords.")

        except FileNotFoundError:
            self.logger.error(f"Event filter config not found at '{config_path}'. The filter will be disabled.")
            # Disable the filter if config is missing to prevent false negatives
            self._risk_actions_regex = self._systemic_entities_regex = self._exclusions_regex = re.compile('a^') # A regex that never matches

    def _compile_regex(self, keywords: Set[str]) -> re.Pattern:
        """Creates a compiled regex pattern from a set of keywords for efficient matching."""
        if not keywords:
            return re.compile('a^') # Return a regex that will never match if the set is empty
        # Create a pattern that matches any of the keywords as whole words (\b)
        # The pattern is case-insensitive (re.IGNORECASE)
        pattern = r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
        return re.compile(pattern, re.IGNORECASE)

    def is_systemic_event(self, news_text: str) -> bool:
        """
        Applies the filtering logic to a given text.

        Args:
            news_text: The news headline or article snippet to analyze.

        Returns:
            True if the event is deemed potentially systemic, False otherwise.
        """
        # 1. Exclusion Logic: Check for negative keywords first for a fast exit.
        if self._exclusions_regex.search(news_text):
            self.logger.debug("Event discarded due to exclusion keyword.")
            return False

        # 2. AND Combination Logic (Core): Must hit both sets.
        has_risk_action = self._risk_actions_regex.search(news_text)
        if not has_risk_action:
            self.logger.debug("Event discarded: No risk action keyword found.")
            return False

        has_systemic_entity = self._systemic_entities_regex.search(news_text)
        if not has_systemic_entity:
            self.logger.debug("Event discarded: No systemic entity keyword found.")
            return False

        # If both conditions are met, it's a candidate for AI analysis.
        self.logger.info("Potentially systemic event identified. Passing for AI analysis.")
        return True
