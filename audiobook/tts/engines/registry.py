"""
TTS Engine Registry

Manages registration and discovery of TTS engines.
"""

from typing import Dict, Type, Optional, List
from audiobook.tts.engines.base import TTSEngine, EngineCapability, VoiceInfo
import importlib
import os


class EngineRegistry:
    """
    Central registry for TTS engines.
    
    Handles engine discovery, registration, and instantiation.
    Acts as a factory for creating engine instances.
    
    Usage:
        # Register an engine
        registry = EngineRegistry()
        registry.register(OrpheusEngine)
        
        # Get an engine instance
        engine = registry.get_engine("orpheus")
        
        # List all engines
        for engine_class in registry.list_engines():
            print(engine_class.name)
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern for global registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._engines: Dict[str, Type[TTSEngine]] = {}
            cls._instance._instances: Dict[str, TTSEngine] = {}
        return cls._instance
    
    def register(self, engine_class: Type[TTSEngine]) -> None:
        """
        Register a TTS engine class.
        
        Args:
            engine_class: The engine class to register (not an instance)
            
        Raises:
            ValueError: If engine name is already registered
            TypeError: If engine_class is not a TTSEngine subclass
        """
        if not isinstance(engine_class, type) or not issubclass(engine_class, TTSEngine):
            raise TypeError(f"Expected TTSEngine subclass, got {type(engine_class)}")
        
        name = engine_class.name
        if name in self._engines:
            print(f"⚠️ Replacing existing engine: {name}")
        
        self._engines[name] = engine_class
        print(f"✅ Registered TTS engine: {engine_class.display_name} ({name})")
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a TTS engine.
        
        Args:
            name: The engine name to unregister
            
        Returns:
            True if engine was unregistered, False if not found
        """
        if name in self._engines:
            del self._engines[name]
            if name in self._instances:
                del self._instances[name]
            return True
        return False
    
    def get_engine(self, name: str, config=None) -> Optional[TTSEngine]:
        """
        Get an engine instance by name.
        
        Creates a new instance if one doesn't exist, or returns cached instance.
        
        Args:
            name: The engine name
            config: Optional EngineConfig for new instances
            
        Returns:
            TTSEngine instance or None if not found
        """
        if name not in self._engines:
            return None
        
        # Return cached instance if exists and no new config provided
        if name in self._instances and config is None:
            return self._instances[name]
        
        # Create new instance
        engine_class = self._engines[name]
        engine = engine_class(config)
        self._instances[name] = engine
        return engine
    
    def get_engine_class(self, name: str) -> Optional[Type[TTSEngine]]:
        """Get the engine class without instantiating."""
        return self._engines.get(name)
    
    def list_engines(self) -> List[Type[TTSEngine]]:
        """Get list of all registered engine classes."""
        return list(self._engines.values())
    
    def list_engine_names(self) -> List[str]:
        """Get list of all registered engine names."""
        return list(self._engines.keys())
    
    def get_engines_by_capability(self, capability: EngineCapability) -> List[Type[TTSEngine]]:
        """Get all engines that support a specific capability."""
        return [
            engine_class for engine_class in self._engines.values()
            if capability in engine_class.capabilities
        ]
    
    def get_all_voices(self) -> Dict[str, List[VoiceInfo]]:
        """
        Get all voices from all registered engines.
        
        Returns:
            Dict mapping engine name to list of VoiceInfo
        """
        all_voices = {}
        for name in self._engines:
            engine = self.get_engine(name)
            if engine:
                all_voices[name] = engine.get_available_voices()
        return all_voices
    
    def get_engine_info_all(self) -> List[Dict]:
        """Get info about all registered engines."""
        info = []
        for name, engine_class in self._engines.items():
            engine = self.get_engine(name)
            if engine:
                info.append(engine.get_engine_info())
            else:
                info.append({
                    "name": engine_class.name,
                    "display_name": engine_class.display_name,
                    "capabilities": [c.name for c in engine_class.capabilities],
                    "initialized": False
                })
        return info
    
    async def initialize_all(self) -> Dict[str, bool]:
        """
        Initialize all registered engines.
        
        Returns:
            Dict mapping engine name to initialization success
        """
        results = {}
        for name in self._engines:
            engine = self.get_engine(name)
            if engine:
                try:
                    results[name] = await engine.initialize()
                except Exception as e:
                    print(f"❌ Failed to initialize {name}: {e}")
                    results[name] = False
        return results
    
    async def shutdown_all(self) -> None:
        """Shutdown all active engine instances."""
        for name, engine in self._instances.items():
            try:
                await engine.shutdown()
                print(f"🔓 Shutdown {name}")
            except Exception as e:
                print(f"⚠️ Error shutting down {name}: {e}")
        self._instances.clear()
    
    def clear(self) -> None:
        """Clear all registered engines (useful for testing)."""
        self._engines.clear()
        self._instances.clear()


# Global registry instance
engine_registry = EngineRegistry()


def register_engine(engine_class: Type[TTSEngine]) -> Type[TTSEngine]:
    """
    Decorator to register an engine class.
    
    Usage:
        @register_engine
        class MyEngine(TTSEngine):
            name = "myengine"
            ...
    """
    engine_registry.register(engine_class)
    return engine_class


def get_engine(name: str) -> Optional[TTSEngine]:
    """Convenience function to get an engine from the global registry."""
    return engine_registry.get_engine(name)


def list_engines() -> List[str]:
    """Convenience function to list all engine names."""
    return engine_registry.list_engine_names()
