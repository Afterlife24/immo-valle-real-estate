import asyncio
import os
from livekit import api
from dotenv import load_dotenv
try:
    from livekit.api.twirp_client import TwirpError  # type: ignore
except Exception:
    TwirpError = Exception  # type: ignore


async def main():
    load_dotenv()

    livekit_api = api.LiveKitAPI()
    try:
        rule = api.SIPDispatchRule(
            dispatch_rule_individual=api.SIPDispatchRuleIndividual(
                room_prefix="call-",
            ),
        )

        agent = api.RoomAgentDispatch(agent_name="inbound_agent")
        room_config = api.RoomConfiguration(agents=[agent])

        request = api.CreateSIPDispatchRuleRequest(
            rule=rule,
            room_config=room_config,
        )

        try:
            await livekit_api.sip.create_sip_dispatch_rule(request)
        except TwirpError as e:
            message = str(e)
            if "Conflicting SIP Dispatch Rules" in message:
                print("SIP dispatch rule already exists or conflicts; skipping creation.")
            else:
                raise
    finally:
        await livekit_api.aclose()


asyncio.run(main())