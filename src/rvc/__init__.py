"""
rvc package — re-exports from rvc_module.py so that:
    from rvc import Config, load_hubert, get_vc, rvc_infer
resolves correctly even though 'rvc' is now a package (directory).
The actual implementation lives in rvc_module.py (renamed from rvc.py
to avoid the Python name collision between src/rvc.py and src/rvc/).

rvc.py was renamed to rvc_module.py to avoid the name clash between
the rvc/ package directory and the old rvc.py wrapper file.
"""

from rvc_module import Config, load_hubert, get_vc, rvc_infer  # noqa: F401

__all__ = ["Config", "load_hubert", "get_vc", "rvc_infer"]
