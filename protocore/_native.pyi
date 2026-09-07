"""Interface of the optional native extension.

The extension ships as a separate distribution, installed by the ``native``
extra, which places this module inside the core's package. The core never
requires it: :mod:`protocore.runtime.token_counting` imports it inside a
``try`` and falls back to its own implementation, which is the specification
for what this module must compute. The stub is here so that the optional
import is typed whether or not the extension is installed.
"""

def estimate_tokens(
    text: str,
    latin: float,
    cyrillic: float,
    cyrillic_json_escape: float,
    cjk: float,
    json_struct: float,
) -> int:
    """Token estimate for ``text`` under the given chars-per-token ratios.

    Ratios are passed as plain numbers so the extension knows nothing about the
    shape of the core's configuration, and a change to that shape can never
    require rebuilding a wheel.
    """
