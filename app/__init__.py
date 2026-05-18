"""MOEX Technical Advisor Agent (advisory only).

The agent NEVER places trades. It only produces structured recommendations
that the main news-driven trading agent can consume via the Python service
class :class:`app.strategy.advisor.TechnicalAdvisor` or via ``POST /advice``.
"""

__version__ = "0.1.0"
