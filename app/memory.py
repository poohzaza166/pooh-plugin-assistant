from bus import MessageBus, Priority


class History:
    def __init__(self, bus: MessageBus):
        self.conversationHistory = []
        self.bus = bus

    def _core_events(self):
        @self.bus.provide_data("history.get_convo")
        def get_user_conversation():
            """Get the full conversation history"""
            return self.conversationHistory

        @self.bus.provide_data("history.latest_conversation")
        def get_latest_covnersation():
            return self.conversationHistory[-1]

        @self.bus.subscribe("intent.query", priority=Priority.HIGH)
        def on_utterance_response(content, **kwargs):
            """Store user input in the conversation history"""
            # print(content)
            self.conversationHistory.append(
                {"content": content, "user": "user"})
            # print(f"Stored message: {content}")

        @self.bus.subscribe("intent.respond", priority=Priority.HIGH)
        def on_plugin_respond(content, **kwargs):
            self.conversationHistory.append(
                {"content": content, "user": "tools"}
            )
