from typing import List, Dict, Optional, Any


from src.utils.logger import get_logger

logger = get_logger(__name__)
class RuleResult:
    """
    A class representing the result of a single rule evaluation.

    Attributes:
        id (str): Unique identifier for the rule
        name (str): Human-readable name of the rule
        rule_type (str): Type/category of the rule
        scope (str): Scope of rule application
        result (str): Raw result string ("Pass" or "Fail")
        latency_ms (int): Rule evaluation latency in milliseconds
        details (Optional[Dict[str, Any]]): Additional rule-specific details
    """

    def __init__(
        self,
        rule_id: str,
        name: str,
        rule_type: str,
        scope: str,
        result: str,
        latency_ms: int,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a RuleResult instance.

        Args:
            rule_id (str): Unique identifier for the rule
            name (str): Human-readable name of the rule
            rule_type (str): Type/category of the rule
            scope (str): Scope of rule application
            result (str): Raw result string ("Pass" or "Fail")
            latency_ms (int): Rule evaluation latency in milliseconds
            details (Optional[Dict[str, Any]], optional): Additional rule-specific details. Defaults to None.
        """
        self.id = rule_id
        self.name = name
        self.rule_type = rule_type
        self.scope = scope
        self.result = result
        self.latency_ms = latency_ms
        self.details = details

    @property
    def result_boolean(self) -> bool:
        """
        Convert the rule result to a boolean value.

        Returns:
            bool: True if the result is "Pass", False otherwise
        """
        return self.result.lower() == "pass"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the RuleResult instance to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing all RuleResult attributes
        """
        return {
            "id": self.id,
            "name": self.name,
            "rule_type": self.rule_type,
            "scope": self.scope,
            "result": self.result_boolean,  # Convert "Pass"/"Fail" to True/False
            "latency_ms": self.latency_ms,
            "details": self.details,
        }


class InferenceResult:
    def __init__(self, input_json: Dict[str, Any]):
        """
        Initializes an InferenceResult object from a JSON dictionary.
        
        :param input_json: The input JSON dictionary.
        """
        self.inference_id = input_json['inference_id']
        self.user_id = input_json['user_id']
        self.rule_results = [
            RuleResult(
                rule_id=rule['id'],
                name=rule['name'],
                rule_type=rule['rule_type'],
                scope=rule['scope'],
                result=rule['result'],
                latency_ms=rule['latency_ms'],
                details=rule.get('details')
            )
            for rule in input_json['rule_results']
        ]

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the InferenceResult object to a dictionary.
        
        :return: A dictionary representation of the object.
        """
        return {
            "inference_id": self.inference_id,
            "user_id": self.user_id,
            "rule_results": [rule.to_dict() for rule in self.rule_results],
        }

    def get_pass_fail_results(self) -> List[Dict[str, Any]]:
        """
        Returns a list of rules with their pass/fail status.
        """
        return [
            {"id": rule.id, "name": rule.name, "result": rule.result_boolean}
            for rule in self.rule_results
        ]

    def get_pass_fail_string(self) -> str:
        """
        Returns a formatted string of rule names and their pass/fail status.
        
        Example output:
            "Content Safety: PASS
             Input Validation: FAIL
             Response Quality: PASS"
        """
        return ", ".join(
            f"{rule.name}: {'PASS' if rule.result_boolean else 'FAIL'}"
            for rule in self.rule_results
        )

    def get_rule_details(self) -> List[Dict[str, Any]]:
        """
        Returns a list of all rules with their details.
        """
        return [
            {"id": rule.id, "name": rule.name, "details": rule.details}
            for rule in self.rule_results
            if rule.details is not None
        ]
    
    def get_inference_id(self) -> str:
        """
        Returns the inference ID.
        """
        return self.inference_id

    def return_pii(self) -> bool:
        """
        Checks if any PII (Personally Identifiable Information) rules exist 
        and returns if they all passed.
        
        Returns:
            bool: True if all PII rules passed or no PII rules found,
                 False if any PII rule failed
        """
        logger.debug("[InferenceResult.return_pii] Checking PII rules")
        if self.rule_results is None:
            logger.debug("[InferenceResult.return_pii] No rules found")
            return True
        for rule in self.rule_results:
            if "pii" in rule.name.lower() or "personally identifiable information" in rule.name.lower():
                logger.debug(f"[InferenceResult.return_pii] Found PII rule: {rule.name}")
                if not rule.result_boolean:
                    logger.info(f"[InferenceResult.return_pii] PII rule failed: {rule.name}")
                    return False
        logger.debug("[InferenceResult.return_pii] All PII rules passed or no PII rules found")
        return True

    def return_hallucination(self) -> bool:
        """
        Checks if any hallucination rules exist 
        and returns if they all passed.
        """ 
        logger.debug("[InferenceResult.return_hallucination] Checking hallucination rules")
        if self.rule_results is None:
            logger.debug("[InferenceResult.return_hallucination] No rules found")
            return True
        for rule in self.rule_results:
            if "hallucination" in rule.name.lower():
                logger.debug(f"[InferenceResult.return_hallucination] Found hallucination rule: {rule.name}")
                if not rule.result_boolean:
                    logger.info(f"[InferenceResult.return_hallucination] Hallucination rule failed: {rule.name}")
                    return False
        logger.debug("[InferenceResult.return_hallucination] All hallucination rules passed or no hallucination rules found")
        return True
