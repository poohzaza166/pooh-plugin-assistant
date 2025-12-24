import datetime
import time

import pytz
from fastapi import APIRouter

from .pooh_lib import (PluginBase, loop_method, provide_data, push_event,
                       register_command, subscribe)

app = APIRouter()

class TimeManagement(PluginBase):
    def __init__(self, message_bus, parser):
        super().__init__(message_bus, parser)
        self.timers = {}
        self.timezone = "UTC"

    @register_command(
        patterns=[r"what(?:'s|s| is) the (?:current )?time", r"time now", r"current time"],
        example_phrase={"time_query": "What's the current time?"}
    )
    def get_current_time(self, format: str = "%H:%M:%S") -> str:
        """Get the current time in the specified format"""
        current_time = datetime.datetime.now().strftime(format)
        return f"The current time is {current_time}"
    
    @register_command(
        patterns=[r"set (?:a|the) timer for (\d+) (second|minute|hour|day)s?", 
                  r"remind me in (\d+) (second|minute|hour|day)s?"],
        example_phrase={"timer": "Set a timer for 5 minutes", 
                       "reminder": "Remind me in 30 minutes"}
    )
    def set_timer(self, duration: int, unit: str, label: str = "Timer") -> str:
        """Set a timer or reminder for the specified duration"""
        # Convert everything to seconds
        unit_multipliers = {
            "second": 1,
            "minute": 60,
            "hour": 3600,
            "day": 86400
        }
        
        seconds = duration * unit_multipliers.get(unit.lower(), 1)
        timer_id = str(time.time())
        end_time = time.time() + seconds
        
        self.timers[timer_id] = {
            "end_time": end_time,
            "label": label,
            "duration": f"{duration} {unit}{'s' if duration > 1 else ''}"
        }
        
        return f"{label} set for {duration} {unit}{'s' if duration > 1 else ''}"
        
    @loop_method(delay=1.0)
    def check_timers(self):
        """Check for completed timers and notify user"""
        current_time = time.time()
        completed_timers = []
        
        for timer_id, timer_info in self.timers.items():
            if current_time >= timer_info["end_time"]:
                completed_timers.append(timer_id)
                self.message_bus.publish("notification", 
                                        f"{timer_info['label']} for {timer_info['duration']} is complete!")
        
        # Remove completed timers
        for timer_id in completed_timers:
            del self.timers[timer_id]
    
    @register_command(
        patterns=[r"convert (\d+:\d+) from (\w+) to (\w+)", 
                  r"what time is (\d+:\d+) (\w+) in (\w+)"]
    )
    def convert_timezone(self, time_str: str, from_zone: str, to_zone: str) -> str:
        """Convert time between different timezones"""
        try:
            # Parse the time string
            time_format = "%H:%M"
            time_obj = datetime.datetime.strptime(time_str, time_format)
            
            # Get timezone objects
            from_tz = pytz.timezone(from_zone)
            to_tz = pytz.timezone(to_zone)
            
            # Set the date to today
            today = datetime.datetime.now().date()
            time_obj = datetime.datetime.combine(today, time_obj.time())
            
            # Localize and convert
            localized_time = from_tz.localize(time_obj)
            converted_time = localized_time.astimezone(to_tz)
            
            return f"{time_str} in {from_zone} is {converted_time.strftime(time_format)} in {to_zone}"
        except (ValueError, pytz.exceptions.UnknownTimeZoneError):
            return f"Sorry, I couldn't convert between {from_zone} and {to_zone}. Please check timezone names."
    
    @register_command(
        patterns=[r"set (?:default )?timezone to (\w+/\w+)"]
    )
    def set_timezone(self, timezone: str) -> str:
        """Set the default timezone for time-related commands"""
        try:
            # Validate timezone
            pytz.timezone(timezone)
            self.timezone = timezone
            return f"Default timezone set to {timezone}"
        except pytz.exceptions.UnknownTimeZoneError:
            return f"Unknown timezone: {timezone}. Please use a valid timezone like 'America/New_York'."
    
    @subscribe("text_input")
    def check_for_time_queries(self, text):
        """Listen for simple time queries that might not match exact commands"""
        time_keywords = ["time", "clock", "hour", "minute", "second"]
        
        if any(keyword in text.lower() for keyword in time_keywords):
            # This is a time-related query, but might not match our patterns exactly
            # We could use a more advanced NLP approach here or just default to current time
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            self.message_bus.publish("response", f"It's currently {current_time}")


def plugin_entry(message_bus, parser, api):
    print("loading plugin")
    print("checking parser object")
    print(parser)
    api.include_router(app, prefix="/time")
    plugin = TimeManagement(message_bus, parser)
    return  plugin
