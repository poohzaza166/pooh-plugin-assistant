
from .pooh_lib import (PluginBase, get_data, loop_method, provide_data,
                       push_event, register_command, subscribe)


def plugin_entry(message_bus, parser, api):
    print("loading plugin")
    print("checking parser object")
    print(parser)
    plugin = GetUserInput(message_bus, parser)
    return plugin


class GetUserInput(PluginBase):
    def __init__(self, message_bus, parser):
        super().__init__(message_bus, parser)
        self.lock = False

    @provide_data("test")
    def return_str(self):
        return "aaaaaaa"

    @get_data("conversation_history")
    def print_latest_conversation(self, obj, **kwargs):
        # print(obj)
        # print(kwargs)
        pass

    @subscribe("text_input")
    def parse_text_input(self, message, **kwargs):
        # print("text input founded")
        # out = self.parser.parse_input(message)
        # if out["confidence"] >= 0.4:
        out = self.parser.execute_command(message)
        print(out)
        self.lock = False
        return

    @loop_method(delay=2)
    def run_publish(self):
        if self.lock:
            return
        user_input = input("please enter new message: ")
        # print(self.input_count)
        self.lock = True
        if user_input != "":
            self.publish("text_input", user_input)
            print(self.print_latest_conversation())

        # ok = self.message_bus.get_data("test")
        # print(ok)
