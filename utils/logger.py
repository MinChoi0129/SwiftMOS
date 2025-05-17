import logging, sys
from datetime import datetime
from pytz import utc, timezone


def config_logger(path: str, is_master: bool = True):
    """멀티-GPU 분산 학습용 로거 설정"""

    def _seoul_time(*args):
        return utc.localize(datetime.utcnow()).astimezone(timezone("Asia/Seoul")).timetuple()

    fmt = "%(asctime)s │ %(levelname)s │ %(message)s"
    datefmt = "%m-%d %H:%M:%S"

    # force=True 로 모든 이전 핸들러 제거
    if is_master:
        logging.basicConfig(
            level=logging.INFO,
            format=fmt,
            datefmt=datefmt,
            handlers=[
                logging.FileHandler(path, mode="a"),  # ← ★ append
                logging.StreamHandler(sys.stdout),
            ],
            force=True,
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=fmt,
            datefmt=datefmt,
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True,
        )

    for h in logging.getLogger().handlers:
        h.setFormatter(logging.Formatter(fmt, datefmt))
        h.formatter.converter = _seoul_time
