"""Extend the model package path so pinned JevLight baseline agents remain importable."""
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
