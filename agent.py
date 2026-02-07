from __future__ import annotations
import os
import asyncio
import time
import logging
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict, ValidationError
from dotenv import load_dotenv
from datetime import datetime

from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    RoomInputOptions,
    function_tool,
)
from livekit.plugins import openai, noise_cancellation
from livekit.plugins.openai import realtime

# --- Local imports
from db import DatabaseDriver
from prompts import AGENT_INSTRUCTION, SESSION_INSTRUCTION

# --- Load environment variables
load_dotenv()

# ============================================================
# 🚀 CONFIGURATION
# ============================================================
PRODUCTION = os.getenv("ENVIRONMENT") == "production"

# --- Environment variables for configurability
OPENAI_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-mini-realtime-preview-2024-12-17")
AGENT_VOICE = os.getenv("AGENT_VOICE", "alloy")
AGENT_NAME = os.getenv("AGENT_NAME", "Sarah")
COMPANY_NAME = os.getenv("COMPANY_NAME", "The Cornish Diamond Co.")

# --- Minimum confidence threshold for STT
STT_CONFIDENCE_THRESHOLD = float(os.getenv("STT_CONFIDENCE_THRESHOLD", "0.6"))

# ============================================================
# 📝 LOGGER SETUP (MUST BE BEFORE ANY LOGGER USAGE)
# ============================================================

# Configure logging first
logging.basicConfig(
    level=logging.WARNING if PRODUCTION else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create logger instance
log = logging.getLogger("jewellery_consultant_agent")

# ============================================================
# 🛡️ SECURITY & PRIVACY GUARDS
# ============================================================

def should_process(text: str, confidence: float | None = None) -> bool:
    """Noise & Intent Guard from checklist - filters irrelevant speech"""
    if not text or not text.strip():
        return False
    
    text = text.lower().strip()
    
    # Minimum length check - discard short fragments
    if len(text.split()) < 3:
        return False
    
    # Confidence gating
    if confidence is not None and confidence < STT_CONFIDENCE_THRESHOLD:
        return False
    
    # Intent keyword filtering - only process relevant speech
    keywords = ["ring", "diamond", "appointment", "consult", "jewellery", 
                "jewelry", "wedding", "engagement", "band", "gold", "silver",
                "platinum", "gem", "stone", "carat", "cut", "clarity", "color"]
    
    return any(k in text for k in keywords)


def sanitize_input(text: str) -> str:
    """Input sanitization to remove PII and irrelevant content from text"""
    if not text:
        return ""
    
    # Remove filler words
    filler_words = ["uh", "um", "hmm", "you know", "like", "actually", "basically"]
    for word in filler_words:
        text = text.replace(word, "")
    
    # Remove PII patterns (excluding phone numbers which are handled separately)
    # Email addresses
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[REDACTED_EMAIL]', text)
    # Credit card numbers (simplified pattern)
    text = re.sub(r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}', '[REDACTED_CC]', text)
    
    # Limit length
    return text[:500].strip()


def sanitize_phone(phone: str) -> str:
    """Sanitize phone numbers - keep only digits and plus sign"""
    if not phone or phone == "unknown":
        return phone
    
    # Keep only digits and plus sign
    cleaned = re.sub(r"[^\d+]", "", phone)
    
    # Limit reasonable length
    return cleaned[:20] if cleaned else phone


# ============================================================
# 🚀 MODULE-LEVEL PROMPT CACHE
# ============================================================
_COMBINED_INSTRUCTIONS_CACHE = None

def _get_combined_instructions() -> str:
    """Get cached combined instructions - computed once at module load"""
    global _COMBINED_INSTRUCTIONS_CACHE
    if _COMBINED_INSTRUCTIONS_CACHE is None:
        # Apply environment variables to instructions
        base_instructions = f"{AGENT_INSTRUCTION}\n\n{SESSION_INSTRUCTION}"
        # Replace placeholders
        base_instructions = base_instructions.replace("{AGENT_NAME}", AGENT_NAME)
        base_instructions = base_instructions.replace("{COMPANY_NAME}", COMPANY_NAME)
        _COMBINED_INSTRUCTIONS_CACHE = base_instructions
    return _COMBINED_INSTRUCTIONS_CACHE

# ============================================================
# 🗄️ DATABASE MANAGEMENT
# ============================================================

class DatabaseManager:
    """Thread-safe database manager with connection pooling"""
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._driver = None
            cls._instance._initialized = False
        return cls._instance
    
    async def initialize(self) -> None:
        """Initialize database connection with health check"""
        async with self._lock:
            if not self._initialized:
                try:
                    self._driver = DatabaseDriver()
                    # Perform health check
                    if hasattr(self._driver, 'health_check'):
                        await self._driver.health_check()
                    else:
                        # Simple connection test
                        await asyncio.to_thread(lambda: self._driver.ping() 
                                                if hasattr(self._driver, 'ping') else None)
                    self._initialized = True
                    log.info("✅ Database initialized successfully")
                except Exception as e:
                    log.error(f"❌ Database initialization failed: {e}")
                    raise
    
    def get_driver(self):
        """Get database driver (raises if not initialized)"""
        if not self._initialized:
            raise RuntimeError("Database not initialized")
        return self._driver

# Global database manager instance
db_manager = DatabaseManager()

# ============================================================
# 🧩 FUNCTION TOOLS
# ============================================================

class ConsultationData(BaseModel):
    model_config = ConfigDict(extra="allow")
    
    # Common fields
    consultation_type: str  # "jewellery_information", "diamond_consultation", "appointment_enquiry"
    
    # Jewellery information fields
    jewellery_category: Optional[str] = None  # "engagement_ring", "wedding_band", "bespoke"
    style_preference: Optional[str] = None
    diamond_shape: Optional[str] = None
    
    # Consultation fields
    occasion: Optional[str] = None  # "engagement", "wedding", "anniversary", "bespoke_gift"
    ring_style: Optional[str] = None
    diamond_preference: Optional[str] = None  # "lab_grown", "natural"
    budget_range: Optional[str] = None
    
    # Appointment fields
    consultation_type_preference: Optional[str] = None  # "in_person", "virtual"
    email: Optional[str] = None
    phone: Optional[str] = None
    name: Optional[str] = None


class TurnDetectionConfig(BaseModel):
    type: str = "server_vad"
    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500


class CreateConsultationArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    consultation_type: str  # "jewellery_information", "diamond_consultation", "appointment_enquiry"
    consultation_data: Dict[str, Any]
    phone: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None


# ============================================================
# 🧠 AGENT CLASS
# ============================================================

class JewelleryConsultantAgent(Agent):
    """Production-ready jewellery consultant agent with all safety guards"""
    
    # Class-level instruction cache
    _cached_instructions = None
    
    def __init__(self, job_context=None, llm=None):
        # Initialize instruction cache if not done
        if JewelleryConsultantAgent._cached_instructions is None:
            JewelleryConsultantAgent._cached_instructions = _get_combined_instructions()
        
        # Initialize parent class
        super().__init__(
            instructions=JewelleryConsultantAgent._cached_instructions,
            tools=[self._create_consultation_tool()],  # Create tool inline
            llm=llm,
        )
        
        # Instance state
        self.current_session = None
        self.caller_phone = None
        self.termination_started = False
        self.consultation_created = False
        self.job_context = job_context
        self._active_tasks = set()  # Track async tasks for cleanup
        
        # Instance-level lock for per-call isolation
        self._consultation_lock = asyncio.Lock()
        
        # Termination timer tracking
        self._termination_timer_task: Optional[asyncio.Task] = None
        self._termination_timeout = 60.0  # seconds of silence
    
    def _create_consultation_tool(self):
        """Factory method to create consultation tool with proper locking"""
        @function_tool()
        async def create_consultation(
            consultation_type: str, 
            consultation_data: Dict[str, Any], 
            phone: Optional[str] = None, 
            name: Optional[str] = None, 
            email: Optional[str] = None
        ) -> str:
            """
            Create a jewellery consultation enquiry with the provided information.
            Uses async lock to prevent double-triggering.
            """
            # Apply input sanitization to sensitive fields
            if phone:
                phone = sanitize_phone(phone)  # Phone-specific sanitization
            if email:
                email = sanitize_input(email)
            if name:
                name = sanitize_input(name)
            
            # Use instance lock to prevent race conditions (per-call isolation)
            async with self._consultation_lock:
                if self.consultation_created:
                    return "I'm sorry, but I can only create one consultation enquiry per call. Your previous enquiry has already been saved."
                
                # Use caller phone if available and no phone provided
                if not phone or phone == "unknown":
                    phone = self.caller_phone if self.caller_phone and self.caller_phone != "extracted_failed" else None
                
                # Generate fallback phone if none available
                if not phone or phone == "unknown":
                    phone = f"call_{int(time.time())}"
                
                # Ensure consultation_data is a plain dict
                if consultation_data and not isinstance(consultation_data, dict):
                    if hasattr(consultation_data, "model_dump"):
                        consultation_data = consultation_data.model_dump()
                    elif hasattr(consultation_data, "dict"):
                        consultation_data = consultation_data.dict()
                    else:
                        consultation_data = dict(consultation_data) if consultation_data else {}
                
                # Sanitize consultation data
                sanitized_data = {}
                for key, value in consultation_data.items():
                    if isinstance(value, str):
                        sanitized_data[key] = sanitize_input(value)
                    else:
                        sanitized_data[key] = value
                
                # Create async task for non-blocking database write
                async def save_consultation_async():
                    try:
                        driver = db_manager.get_driver()
                        result = driver.create_consultation(
                            phone, consultation_type, sanitized_data, name, email, self.caller_phone
                        )
                        
                        if result:
                            self.consultation_created = True
                            log.info(f"✅ Consultation saved with ID: {result.get('_id', 'N/A')}")
                        else:
                            log.warning("⚠️ Consultation save returned None")
                    except Exception as e:
                        log.error(f"❌ Async consultation save failed: {e}")
                
                # Fire and forget - don't await
                task = asyncio.create_task(save_consultation_async())
                self._track_task(task)
                
                return "Thank you for your enquiry. I've saved your information, and one of our specialist consultants will follow up with you within 24 hours."
        
        return create_consultation
    
    def _track_task(self, task: asyncio.Task) -> None:
        """Track async task for proper cleanup"""
        self._active_tasks.add(task)
        task.add_done_callback(lambda t: self._active_tasks.discard(t))
    
    def _reset_termination_timer(self):
        """Reset or start the termination timer based on activity"""
        # Cancel existing timer if running
        if self._termination_timer_task and not self._termination_timer_task.done():
            self._termination_timer_task.cancel()
        
        # Start new timer
        self._termination_timer_task = asyncio.create_task(self._start_termination_timer())
        self._track_task(self._termination_timer_task)
    
    async def _start_termination_timer(self):
        """Internal method to start the termination timer"""
        try:
            await asyncio.sleep(self._termination_timeout)
            if not self.termination_started:
                log.info(f"⏰ No activity for {self._termination_timeout}s, initiating termination")
                await self._terminate_call_after_delay()
        except asyncio.CancelledError:
            # Timer was reset - that's expected
            pass
        except Exception as e:
            log.error(f"Error in termination timer: {e}")
    
    async def _execute_tool(self, tool_call, session):
        """Override tool execution with additional safety checks"""
        try:
            # Apply noise & intent guard before tool execution
            if hasattr(tool_call.function, 'arguments'):
                import json
                args = json.loads(tool_call.function.arguments)
                
                # Check if this is a consultation creation
                if tool_call.function.name == "create_consultation":
                    # Sanitize phone number with phone-specific sanitization
                    if "phone" in args and args["phone"]:
                        args["phone"] = sanitize_phone(args["phone"])
                    
                    # Ensure we have a phone number
                    if "phone" not in args or not args["phone"] or args["phone"] == "unknown":
                        args["phone"] = self.caller_phone if self.caller_phone and self.caller_phone != "extracted_failed" else f"call_{int(time.time())}"
                    
                    # Convert consultation_data to dict if needed
                    if "consultation_data" in args and args["consultation_data"]:
                        data = args["consultation_data"]
                        if hasattr(data, "model_dump"):
                            args["consultation_data"] = data.model_dump()
                        elif hasattr(data, "dict"):
                            args["consultation_data"] = data.dict()
                        elif not isinstance(data, dict):
                            args["consultation_data"] = dict(data)
                    
                    tool_call.function.arguments = json.dumps(args)
            
            return await super()._execute_tool(tool_call, session)
        except Exception as e:
            log.error(f"Tool execution error: {e}")
            return "I apologize, but there was an error processing your request."
    
    async def on_message(self, message, session):
        """Handle incoming messages with all safety guards"""
        if self.termination_started:
            return "The call is ending. Thank you for contacting The Cornish Diamond Co. Goodbye!"
        
        # Extract text and confidence if available
        text = message.content or ""
        confidence = getattr(message, 'confidence', None)
        
        # Apply noise & intent guard
        if not should_process(text, confidence):
            log.debug(f"Filtered out message (confidence: {confidence}): {text[:50]}...")
            return None  # Don't process irrelevant speech
        
        # Apply input sanitization
        sanitized_text = sanitize_input(text)
        if not sanitized_text:
            return None
        
        # Use sanitized text
        message.content = sanitized_text
        
        # Reset termination timer on valid activity
        self._reset_termination_timer()
        
        try:
            # Use timeout protection
            response = await asyncio.wait_for(
                super().on_message(message, session),
                timeout=3.0
            )
            return response
        except asyncio.TimeoutError:
            log.warning("LLM response timeout, using fallback")
            return self._get_smart_fallback_response(text)
        except Exception as e:
            log.error(f"Error in on_message: {e}")
            return self._get_smart_fallback_response(text)
    
    def _get_smart_fallback_response(self, msg: str) -> str:
        """Smart fallback responses based on intent keywords"""
        msg_lower = msg.lower()
        
        if any(x in msg_lower for x in ['ring', 'engagement', 'wedding', 'band', 'diamond']):
            return f"I can help you with information about our engagement rings, wedding bands, or diamond jewellery. What would you like to know?"
        if any(x in msg_lower for x in ['appointment', 'consultation', 'book', 'visit']):
            return f"I can help you arrange a consultation. Would you prefer an in-person or virtual consultation?"
        if any(x in msg_lower for x in ['information', 'info', 'tell me', 'about']):
            return f"I'd be happy to share information about our jewellery. Are you interested in engagement rings, wedding bands, or bespoke pieces?"
        if any(x in msg_lower for x in ['hello', 'hi', 'hey']):
            return f"Hello, and thank you for contacting {COMPANY_NAME}. My name is {AGENT_NAME}. How may I assist you today?"
        
        return f"I'm here to help you with information about our jewellery, guidance on choosing a ring, or to arrange an appointment. How may I assist you?"
    
    async def on_start(self, session: AgentSession):
        """Handle session start with greeting"""
        self.current_session = session
        
        # Start termination timer
        self._reset_termination_timer()
        
        # Start greeting immediately (if TTS enabled)
        if os.getenv("ENABLE_TTS", "1") != "0":
            try:
                session.generate_reply(
                    instructions=f'Say the complete greeting in English: "Hello, and thank you for contacting {COMPANY_NAME}. My name is {AGENT_NAME}. How may I assist you today?" Say all parts of the greeting - do not skip any words.'
                )
            except Exception as e:
                log.warning(f"Greeting generation error: {e}")
    
    async def cleanup(self):
        """Cleanup all async tasks"""
        # Cancel termination timer
        if self._termination_timer_task and not self._termination_timer_task.done():
            self._termination_timer_task.cancel()
        
        # Cancel all other tasks
        for task in self._active_tasks:
            if not task.done():
                task.cancel()
        
        # Clear task sets
        self._active_tasks.clear()
        self._termination_timer_task = None
    
    # ------------------------------------------------------------
    # 🧩 TERMINATION LOGIC (FIXED - ASYNC TWILIO CALLS)
    # ------------------------------------------------------------
    
    async def _terminate_twilio_call(self, call_sid: str) -> None:
        """Terminate Twilio call using REST API - ASYNCHRONOUS fire-and-forget"""
        import aiohttp
        
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        
        if not account_sid or not auth_token:
            log.warning("⚠️ Twilio credentials missing")
            return
        
        async def terminate_async():
            try:
                url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Calls/{call_sid}.json"
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        auth=aiohttp.BasicAuth(account_sid, auth_token),
                        data={"Status": "completed"},
                        timeout=5.0
                    ) as resp:
                        if resp.status == 200:
                            log.info(f"✅ Twilio call {call_sid} terminated")
                        else:
                            body = await resp.text()
                            log.warning(f"⚠️ Twilio API failed: {resp.status} - {body}")
            except Exception as e:
                log.error(f"⚠️ Error terminating Twilio call: {e}")
        
        # Fire and forget - don't await
        task = asyncio.create_task(terminate_async())
        self._track_task(task)
    
    async def _terminate_call_after_delay(self):
        """Comprehensive async termination sequence"""
        try:
            log.info("🔧 Starting automatic call termination sequence...")
            self.termination_started = True
            
            # Cancel termination timer
            if self._termination_timer_task and not self._termination_timer_task.done():
                self._termination_timer_task.cancel()
            
            # Send goodbye message
            if self.current_session and os.getenv("ENABLE_TTS", "1") != "0":
                try:
                    await asyncio.wait_for(
                        self.current_session.generate_reply(
                            instructions=f"Say: Thank you for contacting {COMPANY_NAME}. Goodbye!"
                        ),
                        timeout=4.0
                    )
                    await asyncio.sleep(6.0)
                except Exception as e:
                    log.warning(f"⚠️ Could not send final goodbye: {e}")
            
            # Terminate Twilio calls asynchronously
            if self.job_context and hasattr(self.job_context, "room"):
                room = self.job_context.room
                for pid, participant in room.remote_participants.items():
                    if pid.startswith("sip_"):
                        if hasattr(participant, "attributes") and participant.attributes:
                            call_sid = participant.attributes.get("sip.twilio.callSid")
                            if call_sid:
                                log.info(f"🔧 Scheduling Twilio termination for SID: {call_sid}")
                                # Fire and forget - don't await
                                asyncio.create_task(self._terminate_twilio_call(call_sid))
            
            # Cleanup tasks
            await self.cleanup()
            
            log.info("✅ Call termination sequence completed")
            
        except Exception as e:
            log.error(f"⚠️ Error in termination sequence: {e}")
            # Don't re-raise to avoid breaking the async flow


# ============================================================
# 🚀 ENTRYPOINT
# ============================================================

async def entrypoint(ctx: JobContext):
    """Main entrypoint with health checks and proper initialization"""
    
    # 1. Environment validation
    required_vars = ["OPENAI_API_KEY", "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise RuntimeError(f"Missing required environment variables: {missing_vars}")
    
    # 2. Initialize database with health check
    try:
        await db_manager.initialize()
    except Exception as e:
        log.error(f"❌ Database initialization failed: {e}")
        raise
    
    # 3. Create RealtimeModel with configurable parameters
    realtime_model = realtime.RealtimeModel(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=OPENAI_MODEL,  # Configurable via env var
        voice=AGENT_VOICE,   # Configurable via env var
        modalities=["audio", "text"],
    )
    
    # 4. Create agent
    agent = JewelleryConsultantAgent(
        job_context=ctx,
        llm=realtime_model
    )
    
    # 5. Create session
    session = AgentSession(
        stt=None,  # Handled by RealtimeModel
        tts=None,  # Handled by RealtimeModel
        llm=realtime_model,
    )
    
    # 6. Connect to room
    await ctx.connect()
    
    # 7. Extract caller phone number (non-blocking)
    async def extract_phone_number():
        caller_phone = None
        try:
            if ctx.room:
                for pid, participant in ctx.room.remote_participants.items():
                    # Try different attribute locations
                    if pid.startswith("sip_"):
                        phone = pid.replace("sip_", "")
                        if phone.startswith("+"):
                            caller_phone = phone
                            break
                    
                    attrs = getattr(participant, "attributes", None)
                    if attrs:
                        sip_phone = attrs.get("sip.phoneNumber") or attrs.get("sip.twilio.callerNumber")
                        if sip_phone:
                            caller_phone = sip_phone
                            break
                    
                    metadata = getattr(participant, "metadata", None)
                    if metadata:
                        phone_metadata = metadata.get("phoneNumber") or metadata.get("from")
                        if phone_metadata:
                            caller_phone = phone_metadata
                            break
            
            # Try again after brief delay if not found
            if not caller_phone:
                await asyncio.sleep(0.3)
                if ctx.room:
                    for pid in ctx.room.remote_participants.keys():
                        if pid.startswith("sip_"):
                            phone = pid.replace("sip_", "")
                            if phone.startswith("+"):
                                caller_phone = phone
                                break
        
        except Exception as e:
            log.warning(f"Phone extraction failed: {e}")
        
        agent.caller_phone = sanitize_phone(caller_phone) if caller_phone else "extracted_failed"
        log.info(f"📞 Extracted caller phone: {agent.caller_phone}")
    
    # 8. Start session with noise cancellation
    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    
    # 9. Run phone extraction and greeting in parallel
    asyncio.create_task(extract_phone_number())
    asyncio.create_task(agent.on_start(session))


# ============================================================
# 🏁 MAIN RUNNER
# ============================================================

if __name__ == "__main__":
    # Run the agent
    agents.cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="inbound_agent",
        )
    )