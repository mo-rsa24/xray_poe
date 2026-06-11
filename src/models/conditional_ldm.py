"""ConditionalLDM — public alias for LDM.

The class is implemented in src/models/ldm.py under the name LDM.
This module exposes the plan-11 import contract:

    from src.models.conditional_ldm import ConditionalLDM
"""

from src.models.ldm import LDM as ConditionalLDM

__all__ = ["ConditionalLDM"]
