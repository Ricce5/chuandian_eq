"""Core model composition primitives."""

from .base_model import BaseModel
from .heads import TaskHead
from .task_model import TaskModel

__all__ = ["BaseModel", "TaskHead", "TaskModel"]
