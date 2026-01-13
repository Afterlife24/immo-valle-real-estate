import asyncio
import os
from typing import List, Optional
from livekit import api
from dotenv import load_dotenv
try:
    # TwirpError is raised by the LiveKit API client on API errors
    from livekit.api.twirp_client import TwirpError  # type: ignore
except Exception:  # pragma: no cover - fallback if import path changes
    TwirpError = Exception  # type: ignore


def _parse_allowed_numbers_from_env() -> Optional[List[str]]:
    allowed = os.getenv("INBOUND_ALLOWED_NUMBERS", "").strip()
    if not allowed:
        return None
    numbers = [n.strip() for n in allowed.split(",") if n.strip()]
    return numbers or None


async def main():
    load_dotenv()

    livekit_api = api.LiveKitAPI()
    try:
        allowed_numbers = _parse_allowed_numbers_from_env()

        inbound_trunk = api.SIPInboundTrunkInfo(
            name="livekit-trunk",
            auth_username=os.getenv("TWIML_USERNAME"),
            auth_password=os.getenv("TWIML_PASSWORD"),
            krisp_enabled=True,
            # Setting allowed_numbers makes the trunk specific and avoids conflicts with existing "<any>" trunks
            allowed_numbers=allowed_numbers,  # type: ignore[attr-defined]
        )

        request = api.CreateSIPInboundTrunkRequest(
            trunk=inbound_trunk,
        )

        try:
            await livekit_api.sip.create_sip_inbound_trunk(request)
        except TwirpError as e:
            message = str(e)
            if "Conflicting inbound SIP Trunks" in message:
                # Trunk already exists or conflicts with a broad "any" trunk; skip creating a duplicate
                print("Inbound SIP trunk already exists or conflicts; skipping creation.")
            else:
                raise
    finally:
        await livekit_api.aclose()


asyncio.run(main())