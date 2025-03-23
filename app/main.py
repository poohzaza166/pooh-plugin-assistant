import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI

from .history import register_history_thing
from .model import Command
from .parser import UnifiedCommandParser
from .plugins import PluginManager

app = FastAPI(debug=True)

async def main():
    parser = UnifiedCommandParser()
    plugin_manager = PluginManager("plugins", parser=parser)
    plugin_manager.load_plugins()
    test = register_history_thing(plugin_manager.message_bus)
    plugin_manager.message_bus.publish("example_event", "Hello from main program!")
    await plugin_manager.run_plugins()

    # print(plugin_manager.parser.commands)

    # plugin_manager.message_bus.publish("text_input", "whats the current time?")
    # plugin_manager.parser.parse_input()
    
    try:
        data = plugin_manager.message_bus.get_data("example_data", param="test")
        print(f"Retrieved data: {data}")
    except KeyError as e:
        print(str(e))
    
    # Trigger an event from a plugin
    plugin_manager.message_bus.trigger_event("custom_event", "Some data")
    
if __name__ == "__main__":
    asyncio.run(main())
   

    # # 2. Define command handlers
    # def set_timer(time: str):
    #     print(f"Timer set for {time}")
    #     # print(kwargs)
        
    # def play_music(genre: str, platform: str):
    #     print(f"Playing {genre} music on {platform}")

    # # 3. Register commands
    # parser.register_command(Command(
    #     name="set_timer",
    #     handler=set_timer,
    #     patterns=[
    #         r"set (?:a|the) timer for (\d+) (?:minutes|hours)",
    #         r"remind me in (\d+) (?:minutes|hours)"
    #     ],
    #     required_entities=["TIME"],
    #     dependency_rules={
    #         "timer": {"dep": "dobj", "head": "set"},
    #         "for": {"dep": "prep", "head": "timer"}
    #     }
    # ))

    # parser.register_command(Command(
    #     name="play_music",
    #     handler=play_music,
    #     dependency_rules={
    #         "play": {"dep": "ROOT"},
    #         "music": {"dep": "dobj", "head": "play"}
    #     }
    # ))

    # # 4. Process input
    # input_text = "set a timmer for 10 minutes"
    # result = parser.parse_input(input_text)
    # print(result)
    # """
    # {
    # 'command': 'play_music',
    # 'handler': <function play_music at ...>,
    # 'parameters': {
    #     'target': 'music',
    #     'action': 'Play',
    #     'genre': 'jazz',  # Extracted via NER
    #     'platform': 'Spotify'  # From preposition
    # },
    # 'confidence': 0.75
    # }
    # """

    # # 5. Execute
    # parser.execute_command(input_text)
