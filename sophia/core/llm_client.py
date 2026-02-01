import os
import google.generativeai as genai
import json
import asyncio
from dataclasses import dataclass

@dataclass
class LLMConfig:
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.1

@dataclass
class LLMConfig:
    # Stable High-Poly model for Class 5 Forensics
    model_name: str = "gemini-1.5-pro"
    temperature: float = 0.1

class GeminiClient:
    def __init__(self):
        # RESILIENCE: Check multiple environment keys
        api_key = (os.getenv("SOPHIA_API_KEY") or 
                   os.getenv("GOOGLE_AI_KEY") or 
                   os.getenv("GOOGLE_API_KEY"))
        
        if not api_key:
            print("[WARNING] No API Key in Env. Attempting to load from .env file...")
            try:
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.getenv("SOPHIA_API_KEY") or os.getenv("GOOGLE_AI_KEY")
            except ImportError:
                print("[ERROR] python-dotenv not installed. Secrets must be in ENV.")
            
        if api_key:
            genai.configure(api_key=api_key)
        else:
            print("[CRITICAL] Station OPHANE_NODE_0 is blinded. No API Key found.")
        
    async def query_json(self, prompt: str, system_prompt: str = None) -> dict:
        """
        Forces Gemini to output strict JSON and separates internal thinking.
        Calibrated to config model for stable sovereign throughput.
        """
        model = genai.GenerativeModel(
            model_name=LLMConfig.model_name,
            generation_config={"response_mime_type": "application/json"}
        )
        
        # Cat 1: Separation of Thought
        thought_directive = "\n[THINKING DIRECTIVE]: Wrap your internal reasoning in <thinking>...</thinking> tags before the final JSON output."
        full_system = f"{system_prompt}{thought_directive}" if system_prompt else thought_directive
        
        full_prompt = f"{full_system}\n\nUSER PROMPT:\n{prompt}"
        
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, lambda: model.generate_content(full_prompt))
            
            # Extract Thinking (O1 simulation)
            raw_text = response.text
            if "<thinking>" in raw_text and "</thinking>" in raw_text:
                thinking = raw_text.split("<thinking>")[1].split("</thinking>")[0]
                print(f"\n  [o1] REASONING CHAIN:\n  {thinking.strip()}\n")
                
                # Strip thinking for JSON parsing
                json_part = raw_text.split("</thinking>")[1].strip()
            else:
                json_part = raw_text
                
            return json.loads(json_part)
        except Exception as e:
            print(f"[GEMINI ADAPTER ERROR] {e}")
            return {"error": str(e)}
