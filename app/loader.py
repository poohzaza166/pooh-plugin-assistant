import importlib
import importlib.util
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import yaml
from bus import MessageBus
from plugin_base import Plugin, PluginMetadata


@dataclass
class PluginConfig:
    """Configuration for a single plugin."""
    name: str
    module: str
    class_name: str
    enabled: bool = True
    config: Dict[str, Any] = None

    def __post_init__(self):
        if self.config is None:
            self.config = {}


class PluginLoader:
    """
    Loads and manages plugins dynamically from a plugins directory.
    """

    def __init__(self, bus: MessageBus, plugins_dir: str = "plugins"):
        """
        Initialize the plugin loader.

        Args:
            bus: The message bus instance
            plugins_dir: Directory containing plugin folders
        """
        self.bus = bus
        self.plugins_dir = Path(plugins_dir)
        self.loaded_plugins: Dict[str, Plugin] = {}
        self.plugin_configs: Dict[str, PluginConfig] = {}
        self.logger = logging.getLogger("plugin_loader")

        # Add plugins directory to Python path for imports
        if str(self.plugins_dir.absolute()) not in sys.path:
            sys.path.insert(0, str(self.plugins_dir.absolute()))

    def load_plugin_config(self, config_path: Path) -> Optional[PluginConfig]:
        """
        Load plugin configuration from YAML file.

        Args:
            config_path: Path to the plugin.yaml file

        Returns:
            PluginConfig object or None if loading fails
        """
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            return PluginConfig(
                name=config_data['name'],
                module=config_data['module'],
                class_name=config_data['class'],
                enabled=config_data.get('enabled', True),
                config=config_data.get('config', {})
            )
        except Exception as e:
            self.logger.error(
                f"Failed to load plugin config from {config_path}: {e}")
            return None

    def discover_plugins(self) -> List[PluginConfig]:
        """
        Discover all plugins in the plugins directory.

        Returns:
            List of discovered plugin configurations
        """
        discovered = []

        if not self.plugins_dir.exists():
            self.logger.warning(
                f"Plugins directory {self.plugins_dir} does not exist")
            return discovered

        # Look for plugin.yaml in each subdirectory
        for plugin_dir in self.plugins_dir.iterdir():
            if not plugin_dir.is_dir():
                continue

            config_path = plugin_dir / "plugin.yaml"
            if not config_path.exists():
                continue

            config = self.load_plugin_config(config_path)
            if config:
                discovered.append(config)
                self.logger.info(f"Discovered plugin: {config.name}")

        return discovered

    def load_plugin_class(self, plugin_config: PluginConfig) -> Optional[Type[Plugin]]:
        """
        Dynamically load a plugin class from a module.

        Args:
            plugin_config: Plugin configuration

        Returns:
            Plugin class or None if loading fails
        """
        try:
            # Import the module
            module_path = self.plugins_dir / \
                plugin_config.name / f"{plugin_config.module}.py"

            if not module_path.exists():
                self.logger.error(f"Module file not found: {module_path}")
                return None

            # Load module spec
            spec = importlib.util.spec_from_file_location(
                plugin_config.module,
                module_path
            )

            if spec is None or spec.loader is None:
                self.logger.error(
                    f"Failed to load spec for {plugin_config.module}")
                return None

            # Load the module
            module = importlib.util.module_from_spec(spec)
            sys.modules[plugin_config.module] = module
            spec.loader.exec_module(module)

            # Get the plugin class
            if not hasattr(module, plugin_config.class_name):
                self.logger.error(
                    f"Class {plugin_config.class_name} not found in module {plugin_config.module}"
                )
                return None

            plugin_class = getattr(module, plugin_config.class_name)

            # Verify it's a Plugin subclass
            if not issubclass(plugin_class, Plugin):
                self.logger.error(
                    f"Class {plugin_config.class_name} is not a subclass of Plugin"
                )
                return None

            return plugin_class

        except Exception as e:
            self.logger.error(
                f"Failed to load plugin class {plugin_config.class_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def load_plugin(self, plugin_config: PluginConfig) -> bool:
        """
        Load and initialize a plugin.

        Args:
            plugin_config: Plugin configuration

        Returns:
            True if plugin loaded successfully, False otherwise
        """
        if not plugin_config.enabled:
            self.logger.info(
                f"Plugin {plugin_config.name} is disabled, skipping")
            return False

        try:
            # Load the plugin class
            plugin_class = self.load_plugin_class(plugin_config)
            if plugin_class is None:
                return False

            # Instantiate the plugin
            plugin_instance = plugin_class(self.bus, plugin_config.config)

            # Initialize the plugin
            await plugin_instance.initialize()

            # Store the plugin
            self.loaded_plugins[plugin_config.name] = plugin_instance
            self.plugin_configs[plugin_config.name] = plugin_config

            metadata = plugin_instance.get_metadata()

            self.logger.info(
                f"Loaded plugin: {metadata.name} v{metadata.version} by {metadata.author}"
            )

            return True

        except Exception as e:
            self.logger.error(
                f"Failed to load plugin {plugin_config.name}: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def load_all_plugins(self) -> int:
        """
        Discover and load all plugins.

        Returns:
            Number of successfully loaded plugins
        """
        configs = self.discover_plugins()
        loaded_count = 0

        for config in configs:
            if await self.load_plugin(config):
                loaded_count += 1

        self.logger.info(f"Loaded {loaded_count}/{len(configs)} plugins")
        return loaded_count

    async def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin.

        Args:
            plugin_name: Name of the plugin to unload

        Returns:
            True if plugin unloaded successfully, False otherwise
        """
        if plugin_name not in self.loaded_plugins:
            self.logger.warning(f"Plugin {plugin_name} is not loaded")
            return False

        try:
            plugin = self.loaded_plugins[plugin_name]
            await plugin.shutdown()
            del self.loaded_plugins[plugin_name]
            del self.plugin_configs[plugin_name]

            self.logger.info(f"Unloaded plugin: {plugin_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to unload plugin {plugin_name}: {e}")
            return False

    async def reload_plugin(self, plugin_name: str) -> bool:
        """
        Reload a plugin.

        Args:
            plugin_name: Name of the plugin to reload

        Returns:
            True if plugin reloaded successfully, False otherwise
        """
        if plugin_name not in self.plugin_configs:
            self.logger.warning(f"Plugin {plugin_name} config not found")
            return False

        config = self.plugin_configs[plugin_name]

        # Unload if currently loaded
        if plugin_name in self.loaded_plugins:
            await self.unload_plugin(plugin_name)

        # Reload the configuration
        config_path = self.plugins_dir / plugin_name / "plugin.yaml"
        new_config = self.load_plugin_config(config_path)

        if new_config is None:
            self.logger.error(f"Failed to reload config for {plugin_name}")
            return False

        # Load the plugin with new config
        return await self.load_plugin(new_config)

    async def unload_all_plugins(self) -> None:
        """Unload all plugins."""
        plugin_names = list(self.loaded_plugins.keys())
        for plugin_name in plugin_names:
            await self.unload_plugin(plugin_name)

    def get_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """
        Get a loaded plugin by name.

        Args:
            plugin_name: Name of the plugin

        Returns:
            Plugin instance or None if not found
        """
        return self.loaded_plugins.get(plugin_name)

    def list_plugins(self) -> List[PluginMetadata]:
        """
        List all loaded plugins.

        Returns:
            List of plugin metadata
        """
        return [plugin.get_metadata() for plugin in self.loaded_plugins.values()]

    def get_all_plugin(self) -> List[Plugin]:
        """
        Get a list of all plugin
        """
        return self.loaded_plugins
