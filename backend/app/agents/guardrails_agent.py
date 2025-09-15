from app.core.guardrails import InputGuardrails, OutputGuardrails
from app.models.schemas import GuardrailsResult
import logging

logger = logging.getLogger(__name__)

class GuardrailsAgent:
    """Agent orchestrating input and output guardrails"""
    
    def __init__(self):
        self.input_guardrails = InputGuardrails()
        self.output_guardrails = OutputGuardrails()
        
    async def validate_input(self, question: str) -> GuardrailsResult:
        """Validate input question"""
        logger.info(f"Validating input question: {question[:100]}...")
        result = await self.input_guardrails.validate_input(question)
        logger.info(f"Input validation result: {result.is_valid}, confidence: {result.confidence}")
        return result
        
    async def validate_output(self, question: str, solution: str) -> GuardrailsResult:
        """Validate output solution"""
        logger.info(f"Validating output solution for question: {question[:50]}...")
        result = await self.output_guardrails.validate_output(question, solution)
        logger.info(f"Output validation result: {result.is_valid}, confidence: {result.confidence}")
        return result