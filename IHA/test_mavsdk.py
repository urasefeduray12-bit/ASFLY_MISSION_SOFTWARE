import asyncio
from mavsdk import System
async def test():
    drone = System()
    try:
        await drone.connect(system_address="udpin://10.94.181.89:14550")
        print("No error on connect call")
    except Exception as e:
        print(f"Error: {e}")
asyncio.run(test())
