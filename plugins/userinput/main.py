
from .pooh_lib import (PluginBase, get_data, loop_method, provide_data,
                       push_event, register_command, subscribe)


def plugin_entry(message_bus, parser):
    print("loading plugin")
    print("checking parser object")
    print(parser)
    plugin = GetUserInput(message_bus, parser)
    return  plugin


class GetUserInput(PluginBase):
    def __init__(self, message_bus, parser):
        super().__init__(message_bus, parser)
        self.input_count = 0
    
    # @provide_data("text_input")
    # def get_user_input(self, param):
    #     print("doing stdsadasdsadaddsdadasdsdasdasdauff")
    #     return "hello world"
    
    @provide_data("test")
    def return_str(self):
        return "aaaaaaa"

    @get_data("conversation_history")
    def print_latest_conversation(self, obj, **kwargs):
        print(obj)
        print(kwargs)

    @loop_method(delay=2)
    def run_publish(self):
        # self.input_count += 1
        # print(self.input_count)
        self.publish("text_input", input("please insert a querry"))
        print(self.print_latest_conversation())
        # ok = self.message_bus.get_data("test")  
        # print(ok)