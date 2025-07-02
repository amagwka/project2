"""Common constants for the UDP action/reward servers."""

from config import get_config

_CFG = get_config().action_server

ARROW_DELAY = _CFG.arrow_delay
WAIT_DELAY = _CFG.wait_delay
NON_ARROW_DELAY = _CFG.non_arrow_delay
ARROW_IDX = _CFG.arrow_idx
WAIT_IDX = _CFG.wait_idx

