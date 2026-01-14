# Import OS module for environment variable access
import os

# Load environment variables from a .env file
from dotenv import load_dotenv

# MongoDB client and error classes
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# Typing helper for optional return values
from typing import Optional, List, Dict, Any
import logging

# Import datetime for timestamps
from datetime import datetime


# ---------- Load .env file and initialize MongoDB URI ----------

# Load environment variables from the .env file into the environment
load_dotenv()

# Retrieve MongoDB URI from environment variables
MONGO_URI = os.getenv("MONGO_URI")

# ---------- MongoDB Connection Setup ----------

try:
    # Initialize MongoDB client with the URI
    if not MONGO_URI:
        raise ValueError("MONGO_URI environment variable not set.")
    
    # Disable MongoDB logs by setting logging level
    import logging
    logging.getLogger("pymongo").setLevel(logging.WARNING)
    logging.getLogger("pymongo.connection").setLevel(logging.WARNING)
    logging.getLogger("pymongo.pool").setLevel(logging.WARNING)
    logging.getLogger("pymongo.serverSelection").setLevel(logging.WARNING)
    
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)

    # Access the 'immo_vallee' database
    db = client["immo_vallee"]

    # Access the 'inquiries' collection within the 'immo_vallee' database
    inquiries_collection = db["inquiries"]
    
    # Store client reference for connection testing
    _mongo_client = client
    _mongo_db = db

except (PyMongoError, ValueError) as e:
    # Re-raise, but also log for visibility
    logging.getLogger("realtime_realestate_agent").error(f"Mongo init failed: {e}")
    raise

# ---------- Inquiry Database Driver Class ----------

class DatabaseDriver:
    def __init__(self):
        # Initialize the collection reference to use in other methods
        self.collection = inquiries_collection
        self.log = logging.getLogger("realtime_realestate_agent")
        self._indexes_created = False
        
        # Test database connection on initialization
        try:
            # Ping the database to verify connection
            _mongo_client.admin.command('ping')
            self.log.info("✅ Database connection verified successfully")
        except Exception as e:
            self.log.warning(f"⚠️ Database connection test failed: {e}")
            # Don't raise - connection might still work, just log the warning
        
        # Don't create indexes here - do it lazily on first use to avoid blocking
    
    def _ensure_indexes(self):
        """Create indexes lazily (only once, non-blocking)"""
        if not self._indexes_created:
            try:
                # Create indexes in background (non-blocking)
                self.collection.create_index("phone", background=True)
                self.collection.create_index("created_at", background=True)
                self._indexes_created = True
            except Exception:
                # Silently ignore - indexes are optional optimization
                pass

    # Create a new inquiry in the MongoDB collection
    def create_inquiry(self, phone: str, inquiry_type: str, inquiry_data: Dict[str, Any], name: str = None, caller_phone: str = None) -> Optional[dict]:
        # Log that DB connection is being triggered
        self.log.info("🔌 Database connection triggered by create_inquiry")
        
        # Ensure indexes exist (lazy, non-blocking)
        self._ensure_indexes()
        
        self.log.info(f"Database: Received phone parameter: {phone}")
        self.log.info(f"Database: Phone parameter type: {type(phone)}")
        self.log.info(f"Database: Phone parameter is None: {phone is None}")
        self.log.info(f"Database: Phone parameter == 'unknown': {phone == 'unknown'}")
        
        # NEVER allow "unknown" phone numbers - always use a fallback
        if not phone or phone == "unknown":
            import time
            phone = f"call_{int(time.time())}"
            self.log.info(f"Database: Replaced 'unknown' with fallback phone: {phone}")
        
        self.log.info(f"Database: Final phone number for inquiry: {phone}")
        
        inquiry = {
            "phone": phone,
            "inquiry_type": inquiry_type,  # "property_search", "sell_property", "estimation", "advice"
            "inquiry_data": inquiry_data,
            "status": "new",
            "created_at": datetime.now().isoformat(),
            "source": "phone_call"
        }
        
        # Add optional fields if provided
        if name:
            inquiry["name"] = name
        
        # Add caller phone number if available
        if caller_phone:
            inquiry["caller_phone"] = caller_phone
            inquiry["phone_source"] = "extracted_from_call"
        else:
            inquiry["phone_source"] = "provided_by_customer"
        
        try:
            self.log.info(f"Database: Inserting inquiry with phone: {inquiry.get('phone')}")
            self.log.info(f"Database: Full inquiry document: {inquiry}")
            # Insert the inquiry document into the MongoDB collection
            result = self.collection.insert_one(inquiry)
            self.log.info(f"Database: Insert result: {result.inserted_id}")
            
            # Add MongoDB ID to inquiry for reference
            inquiry["_id"] = str(result.inserted_id)
            
            return inquiry
        except PyMongoError as e:
            self.log.error(f"Database: Insert failed: {e}")
            return None
    
    # Retrieve an inquiry document by phone number
    def get_inquiry_by_phone(self, phone: str) -> Optional[dict]:
        try:
            # Search for an inquiry with the matching phone number, get the most recent one
            inquiry = self.collection.find_one({"phone": phone}, sort=[("_id", -1)])
            return inquiry
        except PyMongoError:
            return None
