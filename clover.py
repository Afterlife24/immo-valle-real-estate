# Clover POS Integration Module
import os
import logging
import aiohttp
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logger
log = logging.getLogger("realtime_realestate_agent")


class CloverClient:
    """
    Clover POS API Client (legacy - not used for real estate).
    
    This module is kept for compatibility but is not used in the real estate agent.
    """
    
    def __init__(
        self,
        merchant_id: str = None,
        access_token: str = None,
        base_url: str = None
    ):
        """
        Initialize Clover API client.
        
        Args:
            merchant_id: Clover merchant ID (defaults to env var)
            access_token: Clover API access token (defaults to env var)
            base_url: Clover API base URL (defaults to env var or sandbox)
        """
        self.merchant_id = merchant_id or os.getenv("CLOVER_MERCHANT_ID")
        self.access_token = access_token or os.getenv("CLOVER_ACCESS_TOKEN")
        self.base_url = base_url or os.getenv(
            "CLOVER_BASE_URL",
            "https://sandbox.dev.clover.com"
        )
        
        if not self.merchant_id:
            log.error("🔍 DEBUG: CLOVER_MERCHANT_ID not found in environment")
            raise ValueError("CLOVER_MERCHANT_ID environment variable not set")
        if not self.access_token:
            log.error("🔍 DEBUG: CLOVER_ACCESS_TOKEN not found in environment")
            raise ValueError("CLOVER_ACCESS_TOKEN environment variable not set")
        
        log.info(f"🔍 DEBUG: Clover client initialized - merchant: {self.merchant_id}, base_url: {self.base_url}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get common headers for API requests."""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
    
    async def create_order(
        self,
        phone: str,
        items: List[Dict[str, Any]],
        name: str = None,
        address: str = None
    ) -> Optional[str]:
        """
        Create a complete order in Clover POS with items.
        
        Args:
            phone: Customer phone number
            items: List of items [{"name": str, "price": float, "quantity": int}]
            name: Customer name (optional)
            address: Customer address (optional)
        
        Returns:
            Clover order ID if successful, None if failed
        """
        try:
            # 🔍 DEBUG: Entry point
            log.info(f"🔍 DEBUG: Clover create_order - phone={phone}, items={items}")
            
            # Step 1: Create the order
            log.info(f"🔍 DEBUG: Creating base order...")
            order_id = await self._create_order_base(phone, name)
            if not order_id:
                log.error("Failed to create base order in Clover")
                return None
            
            log.info(f"🔍 DEBUG: Base order created: {order_id}")
            
            # Step 2: Add line items to the order
            log.info(f"🔍 DEBUG: Adding {len(items)} items to order...")
            success = await self._add_line_items(order_id, items)
            if not success:
                log.error(f"Failed to add items to Clover order {order_id}")
                return None
            
            log.info(f"🔍 DEBUG: Items added successfully")
            log.info(f"✅ Clover order complete: {order_id}")
            
            # Step 3: (Optional) Fire order to kitchen - uncomment if needed
            # await self._fire_order(order_id)
            
            return order_id
            
        except Exception as e:
            log.error(f"Clover order creation failed: {e}")
            import traceback
            log.error(f"🔍 DEBUG: Clover traceback: {traceback.format_exc()}")
            return None
    
    async def _create_order_base(
        self,
        phone: str,
        name: str = None
    ) -> Optional[str]:
        """
        Create base order in Clover.
        
        Args:
            phone: Customer phone number
            name: Customer name
        
        Returns:
            Order ID if successful, None otherwise
        """
        url = f"{self.base_url}/v3/merchants/{self.merchant_id}/orders"
        
        # Build order title
        title = f"Phone Order"
        if name:
            title += f" - {name}"
        
        # Build order note with phone number
        note = f"Phone: {phone}"
        
        payload = {
            "state": "open",
            "title": title,
            "note": note
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("id")
                    else:
                        error_text = await response.text()
                        log.error(f"Clover API error ({response.status}): {error_text}")
                        return None
        except Exception as e:
            log.error(f"Error creating Clover order: {e}")
            return None
    
    async def _add_line_items(
        self,
        order_id: str,
        items: List[Dict[str, Any]]
    ) -> bool:
        """
        Add line items (dishes) to an existing order.
        
        Args:
            order_id: Clover order ID
            items: List of items with name, price, quantity
        
        Returns:
            True if successful, False otherwise
        """
        url = f"{self.base_url}/v3/merchants/{self.merchant_id}/orders/{order_id}/bulk_line_items"
        
        # Convert items to Clover format
        # Clover expects:
        # - Prices in CENTS (multiply by 100)
        # - Quantities in units of 1000 (multiply by 1000)
        clover_items = []
        for item in items:
            clover_items.append({
                "name": item["name"],
                "price": int(item["price"] * 100),  # Convert dollars to cents
                "unitQty": int(item["quantity"] * 1000)  # Convert to Clover unit format
            })
        
        payload = {"items": clover_items}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        return True
                    else:
                        error_text = await response.text()
                        log.error(f"Clover add items error ({response.status}): {error_text}")
                        return False
        except Exception as e:
            log.error(f"Error adding items to Clover order: {e}")
            return False
    
    async def _fire_order(self, order_id: str) -> bool:
        """
        Fire order to kitchen (optional - sends order to kitchen display).
        
        Args:
            order_id: Clover order ID
        
        Returns:
            True if successful, False otherwise
        """
        url = f"{self.base_url}/v3/merchants/{self.merchant_id}/orders/{order_id}/fire"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        log.info(f"Order {order_id} fired to kitchen")
                        return True
                    else:
                        error_text = await response.text()
                        log.warning(f"Failed to fire order ({response.status}): {error_text}")
                        return False
        except Exception as e:
            log.warning(f"Error firing order to kitchen: {e}")
            return False
    
    async def get_merchant_info(self) -> Optional[Dict[str, Any]]:
        """
        Get merchant information (useful for testing connection).
        
        Returns:
            Merchant data if successful, None otherwise
        """
        url = f"{self.base_url}/v3/merchants/{self.merchant_id}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        log.error(f"Failed to get merchant info ({response.status}): {error_text}")
                        return None
        except Exception as e:
            log.error(f"Error getting merchant info: {e}")
            return None


# Singleton instance (optional - for easy access)
_clover_client = None


def get_clover_client() -> CloverClient:
    """
    Get or create singleton Clover client instance.
    
    Returns:
        CloverClient instance
    """
    global _clover_client
    if _clover_client is None:
        _clover_client = CloverClient()
    return _clover_client

