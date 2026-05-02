from enum import Enum
from datetime import datetime

from ok import TriggerTask


class AutoCoffe45sTask(TriggerTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_config = {"_enabled": False}
        self.name = "自动店长特供2-8关卡抢钱"
        self.description = "注意每次使用前要手动通关一次2-8以确保该关卡是默认选择的关卡，要带娜娜莉，水月和白藏可选. PS: 任务终端重开请手动关闭开启这个任务来刷新状态"
        self.state = StateEnum.START
        self.level_start_time = None


    def run(self):
        if self.state == StateEnum.START:
            if self.ocr(0.623, 0.502, 0.675, 0.531, "店长特供"):
                self.send_key('f')
                self.state = StateEnum.SELECT
            return
        if self.state == StateEnum.SELECT:
            if self.ocr(0.864, 0.915, 0.928, 0.946, "开始营业"):
                self.click_relative(0.893, 0.931)
                self.state = StateEnum.WAIT
            return
        if self.state == StateEnum.WAIT:
            if self.ocr(0.800, 0.094, 0.844, 0.123, "营业额"):
                self.level_start_time = datetime.now()
                self.state = StateEnum.INGAME
            return
        if self.state == StateEnum.INGAME:
            self.click_relative(0.049, 0.414)
            now = datetime.now()
            if (now - self.level_start_time).total_seconds() > 48:
                self.state = StateEnum.SETTLE
            self.sleep(0.1)
            return
        if self.state == StateEnum.SETTLE:
            if self.ocr(0.587, 0.762, 0.627, 0.792, "领取"):
                self.click_relative(0.605, 0.775)
                self.state = StateEnum.START
            return

    def start(self):
        self.executor.start()
        self.enable()

    def disable(self) -> None:
        super().disable()
        self.on_disabled()

    def on_disabled(self):
        self.state = StateEnum.START

class StateEnum(Enum):
    START = 0,
    SELECT = 1,
    WAIT = 2,
    INGAME = 3,
    SETTLE = 4