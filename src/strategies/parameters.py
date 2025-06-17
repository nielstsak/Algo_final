# src/strategies/parameters.py

"""
Provides a unified and extensible system for strategy parameter management.

This module replaces the previous Pydantic-based configuration with a more
flexible and descriptive system. It allows strategies to declare their
parameters in a structured way, enabling automatic validation, serialization,
and generation of optimization search spaces.

Key Components:
- Parameter (ABC): The base class for any type of parameter.
- IntParameter, FloatParameter, CategoricalParameter, BoolParameter: Concrete
  implementations for common parameter types.
- ParameterSet: A container for a collection of Parameters, representing the
  complete configuration for a strategy.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, TypeVar, Union, Sequence, Optional

from src.core.exceptions import StrategyInvalidParameterError, SearchSpaceError

T = TypeVar('T')


class Parameter(ABC, Generic[T]):
    """
    Abstract base class for a strategy parameter.

    Each parameter has a name, a default value, and a help string for
    documentation.
    """

    def __init__(self, name: str, default: T, help: str):
        """
        Initializes a Parameter.

        Args:
            name (str): The name of the parameter.
            default (T): The default value for the parameter.
            help (str): A description of the parameter's purpose.
        """
        self.name = name
        self.default = default
        self.help = help

    @abstractmethod
    def validate(self, value: Any) -> T:
        """
        Validates a given value against the parameter's type and constraints.

        Args:
            value (Any): The value to validate.

        Returns:
            T: The validated value, potentially cast to the correct type.

        Raises:
            StrategyInvalidParameterError: If the value is invalid.
        """
        raise NotImplementedError

    @abstractmethod
    def to_optuna_spec(self) -> Dict[str, Any]:
        """
        Generates the specification for this parameter for Optuna trial suggestions.

        Returns:
            Dict[str, Any]: A dictionary containing the type and range for Optuna.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', default={self.default})"


class FloatParameter(Parameter[float]):
    """A parameter whose value is a float."""

    def __init__(self, name: str, default: float, min_value: float, max_value: float, help: str):
        super().__init__(name, default, help)
        if min_value >= max_value:
            raise ValueError("min_value must be less than max_value")
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, value: Any) -> float:
        try:
            float_value = float(value)
        except (ValueError, TypeError) as e:
            raise StrategyInvalidParameterError(
                f"Parameter '{self.name}' must be a float, but got type {type(value)}."
            ) from e

        if not (self.min_value <= float_value <= self.max_value):
            raise StrategyInvalidParameterError(
                f"Value for '{self.name}' ({float_value}) is out of the allowed "
                f"range [{self.min_value}, {self.max_value}]."
            )
        return float_value

    def to_optuna_spec(self) -> Dict[str, Any]:
        return {"type": "float", "low": self.min_value, "high": self.max_value}


class IntParameter(Parameter[int]):
    """A parameter whose value is an integer."""

    def __init__(self, name: str, default: int, min_value: int, max_value: int, step: int = 1, help: str = ""):
        super().__init__(name, default, help)
        if min_value >= max_value:
            raise ValueError("min_value must be less than max_value")
        self.min_value = min_value
        self.max_value = max_value
        self.step = step

    def validate(self, value: Any) -> int:
        if not isinstance(value, int) or isinstance(value, float) and value != int(value):
            try:
                # Attempt to convert, but only if it's a whole number
                if float(value) != int(value):
                    raise ValueError
                value = int(value)
            except (ValueError, TypeError) as e:
                 raise StrategyInvalidParameterError(
                    f"Parameter '{self.name}' must be an integer, but got {value}."
                ) from e

        if not (self.min_value <= value <= self.max_value):
            raise StrategyInvalidParameterError(
                f"Value for '{self.name}' ({value}) is out of the allowed "
                f"range [{self.min_value}, {self.max_value}]."
            )
        return value

    def to_optuna_spec(self) -> Dict[str, Any]:
        return {"type": "int", "low": self.min_value, "high": self.max_value, "step": self.step}


class CategoricalParameter(Parameter[T]):
    """A parameter whose value must be one of a predefined set of choices."""

    def __init__(self, name: str, default: T, choices: Sequence[T], help: str = ""):
        if not choices:
            raise ValueError("Choices cannot be empty.")
        if default not in choices:
            raise ValueError(f"Default value '{default}' is not in the provided choices.")
        super().__init__(name, default, help)
        self.choices = choices

    def validate(self, value: Any) -> T:
        if value not in self.choices:
            raise StrategyInvalidParameterError(
                f"Value for '{self.name}' ({value}) is not in the allowed "
                f"choices: {self.choices}."
            )
        return value

    def to_optuna_spec(self) -> Dict[str, Any]:
        return {"type": "categorical", "choices": list(self.choices)}


class BoolParameter(Parameter[bool]):
    """A parameter whose value is a boolean."""

    def validate(self, value: Any) -> bool:
        if not isinstance(value, bool):
            raise StrategyInvalidParameterError(
                f"Parameter '{self.name}' must be a boolean, but got type {type(value)}."
            )
        return value

    def to_optuna_spec(self) -> Dict[str, Any]:
        return {"type": "categorical", "choices": [True, False]}


class ParameterSet:
    """A collection of parameters that defines a strategy's configuration."""

    def __init__(self, params: Optional[List[Parameter]] = None):
        self._params: Dict[str, Parameter] = {p.name: p for p in params} if params else {}

    def add(self, param: Parameter) -> None:
        """Adds a parameter to the set."""
        if param.name in self._params:
            raise ValueError(f"Parameter with name '{param.name}' already exists.")
        self._params[param.name] = param

    def get_param(self, name: str) -> Optional[Parameter]:
        """Retrieves a parameter by its name."""
        return self._params.get(name)

    def get_defaults(self) -> Dict[str, Any]:
        """Returns a dictionary of all default parameter values."""
        return {name: p.default for name, p in self._params.items()}

    def validate(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates a dictionary of values against the parameter set.

        Args:
            values (Dict[str, Any]): A dictionary of parameter names to values.

        Returns:
            Dict[str, Any]: The dictionary of validated and correctly-typed values.

        Raises:
            StrategyInvalidParameterError: If any value is invalid or a required
                                         parameter is missing.
        """
        validated_values = {}
        for name, param in self._params.items():
            if name not in values:
                raise StrategyInvalidParameterError(f"Required parameter '{name}' is missing.")
            
            value = values[name]
            try:
                validated_values[name] = param.validate(value)
            except StrategyInvalidParameterError as e:
                # Re-raise with more context if needed, or just let it propagate.
                raise e
        
        # Check for extraneous parameters
        for name in values:
            if name not in self._params:
                 raise StrategyInvalidParameterError(f"Unknown parameter '{name}' provided.")
        
        return validated_values

    def generate_optuna_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Generates a dictionary defining the search space for Optuna.

        Returns:
            Dict[str, Dict[str, Any]]: A mapping of parameter names to their
                                       Optuna specification.
        """
        space = {}
        for name, param in self._params.items():
            try:
                space[name] = param.to_optuna_spec()
            except Exception as e:
                raise SearchSpaceError(
                    f"Failed to generate Optuna space for parameter '{name}'"
                ) from e
        return space

    def __iter__(self):
        return iter(self._params.values())

    def __len__(self) -> int:
        return len(self._params)

    def __repr__(self) -> str:
        return f"ParameterSet(params={list(self._params.values())})"
