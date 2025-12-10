from mindspore.profiler import schedule

class Schedule:
    def __init__(self, wait: int, active: int, warmup: int = 0, repeat: int = 0, skip_first: int = 0) -> None:
        self.scheduler = schedule(
            wait=wait,
            active=active,
            warmup=warmup,
            repeat=repeat,
            skip_first=skip_first
        )
