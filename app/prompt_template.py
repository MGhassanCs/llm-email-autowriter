"""
Prompt templates for email generation with different tones and contexts.
Based on patterns from the prototype notebook.
"""
from typing import Dict, List, Any
from .config import Config

class EmailPromptTemplate:
    """Handles prompt generation for email writing tasks"""
    
    def __init__(self):
        self.config = Config()
    
    def _get_tone_instructions(self, tone: str) -> str:
        """Get tone-specific instructions"""
        tone_instructions = {
            "Professional": "Write in a professional, business-appropriate tone. Use formal language and proper email etiquette.",
            "Friendly": "Write in a warm, friendly tone while maintaining professionalism. Be approachable and personable.",
            "Formal": "Write in a very formal, official tone. Use sophisticated language and traditional email structure.",
            "Casual": "Write in a casual, relaxed tone but still appropriate for email communication.",
            "Polite": "Write in a polite, courteous tone. Be respectful and considerate in your language."
        }
        return tone_instructions.get(tone, tone_instructions["Professional"])
    
    def _get_length_instructions(self, length: str) -> str:
        """Get length-specific instructions"""
        length_instructions = {
            "Short": "Keep the email concise and to the point. Aim for 1-2 short paragraphs.",
            "Medium": "Write a moderately detailed email with 2-3 paragraphs providing necessary context.",
            "Long": "Write a comprehensive email with multiple paragraphs, detailed explanations, and thorough context."
        }
        return length_instructions.get(length, length_instructions["Medium"])
    
    def is_email_related(self, intent: str) -> bool:
        """Check if the user intent is email-related"""
        email_keywords = [
            "email", "mail", "send an email", "write an email", "compose email",
            "message", "contact", "reach out", "correspondence", "letter"
        ]
        return any(keyword in intent.lower() for keyword in email_keywords)
    
    def generate_system_prompt(self, tone: str = "Professional", length: str = "Medium") -> str:
        """Generate system prompt based on tone and length"""
        tone_instruction = self._get_tone_instructions(tone)
        length_instruction = self._get_length_instructions(length)
        
        return f"""
        You are a helpful email writing assistant. You specialize in writing professional, well-structured emails based on user intents.

Instructions:
- {tone_instruction}
- {length_instruction}
- Always include a clear subject line
- Use proper email structure (greeting, body, closing, signature placeholders)
- Make the email contextually appropriate and actionable
- Fill in placeholder information with [brackets] when specific details are needed
- Ensure the email is complete and ready to send

Write only the email content without any additional commentary or explanations."""
    
    def generate_non_email_response(self) -> List[Dict[str, str]]:
        """Generate response for non-email related queries"""
        return [
            {
                "role": "system",
                "content": (
                    "You are a strict assistant that only helps with writing emails. "
                    "If a user gives a prompt that is not related to writing an email, "
                    "you will politely ask them to provide a clear instruction to write an email."
                )
            }
        ]
    
    def generate_email_prompt(self, intent: str, tone: str = "Professional", length: str = "Medium") -> List[Dict[str, str]]:
        """Generate the complete prompt for email writing"""
        if not self.is_email_related(intent):
            return self.generate_non_email_response() + [
                {
                    "role": "user",
                    "content": intent
                }
            ]
        
        system_prompt = self.generate_system_prompt(tone, length)
        
        # Enhanced user prompt with context
        user_prompt = f"""Please write an email based on the following intent: {intent}

Additional context:
- Tone: {tone}
- Length: {length}
- Include subject line
- Use professional email structure
- Add placeholder brackets [like this] for specific information that needs to be filled in"""
        
        return [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": user_prompt
            }
        ]
    
    def get_example_intents(self) -> List[str]:
        """Return example email intents for the UI"""
        return [
            "Ask professor for deadline extension due to personal issues",
            "Request meeting with manager to discuss project timeline",
            "Thank colleague for their help on recent presentation",
            "Follow up on job application submitted last week",
            "Apologize for missing the team meeting yesterday",
            "Invite team members to project kickoff meeting",
            "Request vacation time approval for next month",
            "Inform client about project status update",
            "Ask HR about health insurance benefits",
            "Schedule lunch meeting with potential business partner"
        ]
