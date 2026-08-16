"""Sphinx configuration — fleet standard via py-canon."""

from py_canon.sphinx import configure

configure(globals())

extensions.append("nbsphinx")
nbsphinx_execute = "always"
nbsphinx_timeout = 600
