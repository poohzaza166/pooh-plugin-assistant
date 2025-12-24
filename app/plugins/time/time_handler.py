from datetime import datetime
from typing import Any, Dict, List, Optional

import pytz
from bus import Priority
from llm import IntentPlugin
from plugin_base import PluginMetadata


class TimePlugin(IntentPlugin):
    """
    Plugin that handles time and date queries.
    """

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="time",
            version="1.0.0",
            description="Provides current time and date information",
            author="Your Name",
            dependencies=[]
        )

    async def initialize(self) -> None:
        """Initialize the time plugin."""
        await super().initialize()

        # print(self.to_claude_format())

        # Get timezone from config
        self.timezone = self.get_config_value("timezone", "UTC")
        self.logger.info(
            f"Time plugin initialized with timezone: {self.timezone}")

        # Subscribe to intent queries
        @self.bus.subscribe("intent.query", priority=Priority.NORMAL)
        async def on_query(data: Dict[str, Any]):
            # print(data)
            utterance = data.get("utterance", "")
            context = data.get("context", {})

            response = await self.handle_intent(utterance, context)
            if response:
                await self.bus.publish_async("intent.response", {
                    "plugin": self.get_metadata().name,
                    "utterance": utterance,
                    "response": response
                })

        # Provide time data to other plugins
        @self.bus.provide_data("time.current")
        async def get_current_time(timezone: str = None) -> Dict[str, Any]:
            if timezone is None:
                timezone = self.timezone

            tz = pytz.timezone(timezone)
            now = datetime.now(tz)

            return {
                "datetime": now.isoformat(),
                "timestamp": now.timestamp(),
                "timezone": timezone,
                "hour": now.hour,
                "minute": now.minute,
                "second": now.second,
                "year": now.year,
                "month": now.month,
                "day": now.day,
                "weekday": now.strftime("%A")
            }

        # Optional: Periodic time announcement
        if self.get_config_value("announce_hourly", False):
            @self.bus.loop_method(delay=3600.0)  # Every hour
            async def hourly_announcement():
                time_data = await self.bus.get_data_async("time.current")
                hour = time_data["hour"]
                self.logger.info(f"Hourly announcement: It's {hour}:00")

                await self.bus.publish_async("tts.speak", {
                    "text": f"It's {hour} o'clock"
                })

    def get_intent_keywords(self) -> List[str]:
        """Return keywords this plugin handles."""
        return [
            "time",
            "clock",
            "date",
            "day",
            "what time",
            "what day",
            "what's the time",
            "what's the date"
        ]

    async def handle_intent(self, utterance: str, context: Dict[str, Any]) -> Optional[str]:
        """
        Handle time and date queries.

        Args:
            utterance: User's input
            context: Additional context

        Returns:
            Response string or None
        """
        utterance_lower = utterance.lower()

        # Check if this is a time/date query
        if not any(keyword in utterance_lower for keyword in self.get_intent_keywords()):
            return None

        # Get timezone from context or use default
        timezone = context.get("timezone", self.timezone)

        # Get current time data
        time_data = await self.bus.get_data_async("time.current", timezone)

        # Determine what information to return
        if any(word in utterance_lower for word in ["time", "clock", "what time"]):
            hour = time_data["hour"]
            minute = time_data["minute"]

            # Convert to 12-hour format
            period = "AM" if hour < 12 else "PM"
            display_hour = hour if hour <= 12 else hour - 12
            if display_hour == 0:
                display_hour = 12

            return f"The current time is {display_hour}:{minute:02d} {period}."

        elif any(word in utterance_lower for word in ["date", "day", "what day"]):
            weekday = time_data["weekday"]
            month = time_data["month"]
            day = time_data["day"]
            year = time_data["year"]

            month_names = [
                "", "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ]

            return f"Today is {weekday}, {month_names[month]} {day}, {year}."

        # Default response
        return f"I can tell you the time or date. Just ask!"

    async def llm_run(self, utterance, context, **kwargs):
        pass

    async def shutdown(self) -> None:
        """Clean up plugin resources."""
        self.logger.info("Time plugin shutting down")
        await super().shutdown()
