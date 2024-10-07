import os
import logging

DEBUG = (
    os.environ.get("DEBUG", "False").lower() == "true"
)  # If Debug==True runs on localhost with no ssl
PORT = int(os.environ.get("PORT", 7860))
SHARE = os.environ.get("SHARE", "False").lower() == "true"
FORCE_CPU = (
    os.environ.get("FORCE_CPU", "False").lower() == "true"
)  # TODO CPU ONLY: NOT SUPPORTED -> float16

SSL_CERTFILE = "certificates/cert.pem"
SSL_KEYFILE = "certificates/key.pem"


def setup_logging() -> None:
    "Centralized logging configuration"
    log_level = logging.DEBUG if DEBUG else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )


setup_logging()

logging.debug(
    f"Configuration: Debug:{DEBUG}, PORT:{PORT}, SHARE:{SHARE}, FORCE_CPU:{FORCE_CPU}"
)

if FORCE_CPU:
    raise NotImplementedError("CPU-only mode is not yet supported in this application.")
