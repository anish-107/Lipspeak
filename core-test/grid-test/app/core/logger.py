''' logger.py
@authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
@description: Centralized logging configuration
for GRID lip reading backend.
@date: 10 May 2026
@returns: Application Logger

'''


# Imports
import logging



# Logger Configuration
logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s - "
        "%(levelname)s - "
        "%(name)s - "
        "%(message)s"
    )
)



# Application Logger
logger: logging.Logger = logging.getLogger(
    "lipspeak"
)