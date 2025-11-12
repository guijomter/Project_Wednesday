import logging
import os
import datetime

print(f"CWD en test_log.py: {os.getcwd()}")
os.makedirs("logs_test", exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG, # Pongamos DEBUG para estar seguros
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs_test/test.log", mode="w"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

logger.debug("Mensaje de DEBUG")
logger.info("Mensaje de INFO")
logger.warning("Mensaje de WARNING")
logger.critical("Mensaje de CRITICAL")

print("Test de log finalizado. Revisa 'logs_test/test.log'")