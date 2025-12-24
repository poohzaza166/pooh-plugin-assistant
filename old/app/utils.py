import inspect
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

# Store decorated functions before message_bus is available
_pending_registrations = []

def subscribe(event_name: str):
    """Mark a method to be subscribed to an event"""
    def decorator(func: Callable):
        setattr(func, '_subscribe_event', event_name)
        _pending_registrations.append(('subscribe', event_name, func))
        return func
    return decorator

def provide_data(data_name: str):
    """Mark a method as a data provider"""
    def decorator(func: Callable):
        setattr(func, '_provide_data', data_name)
        _pending_registrations.append(('provide_data', data_name, func))
        return func
    return decorator

def loop_method(delay: float = 0):
    """Mark a method to run in a loop"""
    def decorator(func: Callable):
        setattr(func, '_loop_method', delay)
        _pending_registrations.append(('loop_method', delay, func))
        return func
    return decorator

def push_event(event_name: str):
    """Mark a method as an event pusher"""
    def decorator(func: Callable):
        setattr(func, '_push_event', event_name)
        _pending_registrations.append(('push_event', event_name, func))
        return func
    return decorator

def register_with_message_bus(instance, message_bus):
    """Register all decorated methods with the message bus"""
    # First register methods in the instance
    for name, func in inspect.getmembers(instance.__class__, predicate=inspect.isfunction):
        bound_method = getattr(instance, name)
        
        if hasattr(func, '_subscribe_event'):
            event_name = getattr(func, '_subscribe_event')
            message_bus.subscribe(event_name)(bound_method)
            
        if hasattr(func, '_provide_data'):
            data_name = getattr(func, '_provide_data')
            message_bus.provide_data(data_name)(bound_method)
            
        if hasattr(func, '_loop_method'):
            delay = getattr(func, '_loop_method')
            message_bus.loop_method(delay)(bound_method)
            
        if hasattr(func, '_push_event'):
            event_name = getattr(func, '_push_event')
            message_bus.push_event(event_name)(bound_method)
    
    # Store message_bus as instance attribute
    instance.message_bus = message_bus
    
    return instance