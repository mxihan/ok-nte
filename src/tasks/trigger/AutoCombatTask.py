import time

from qfluentwidgets import FluentIcon

from ok import TriggerTask, Logger
from src.combat.BaseCombatTask import BaseCombatTask, NotInCombatException, CharDeadException
from PySide6.QtCore import Signal, QObject
from src.char.CharFactory import get_char_feature_by_pos
from src.char.custom.CustomCharManager import CustomCharManager

class ScannerSignals(QObject):
    # Sends list of dicts: {"index": i, "feat_id": tmp_id, "mat": ndarray, "match": str|None}
    scan_done = Signal(list)

scanner_signals = ScannerSignals()

logger = Logger.get_logger(__name__)


class AutoCombatTask(BaseCombatTask, TriggerTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_config = {'_enabled': True}
        self.trigger_interval = 0.1
        self.name = "自动战斗"
        self.icon = FluentIcon.CALORIES
        self.last_is_click = False
        self.default_config.update({
            '自动目标': True,
        })
        self.config_description = {
            '自动目标': '关闭以仅在手动使用中键选择敌人时启用自动战斗',
        }
        self.op_index = 0
        self.origin_func = {}

    def run(self):
        ret = False
        if not self.scene.in_team(self.in_team_and_world):
            return
        
        combat_start = time.time()
        while self.in_combat():
            ret = True
            try:
                self.get_current_char().perform()
            except CharDeadException:
                self.log_error('Characters dead', notify=True)
                break
            except NotInCombatException as e:
                logger.info(f'auto_combat_task_out_of_combat {int(time.time() - combat_start)} {e}')
                break
        if ret:
            self.combat_end()

    def scan_team(self):
        self.log_info("开始扫描当前队伍...")
        in_team, _, count = self.in_team()
        if not in_team or count == 0:
            scanner_signals.scan_done.emit([])
            self.log_info("队伍不存在，扫描结束")
            return

        manager = CustomCharManager()
        results = []
        frame = self.frame
        is_single = count == 1
        for i in range(count):
            big_mat, w, h = get_char_feature_by_pos(self, i, frame=frame, scale_box=1.1, single_char=is_single)
            if big_mat is not None and big_mat.size > 0:
                is_match, match_name, _ = manager.match_feature(big_mat, threshold=0.8)
                feature_mat = get_char_feature_by_pos(self, i, frame=frame, single_char=is_single)[0]
                results.append({
                    "index": i,
                    "mat": feature_mat,
                    "width": w,
                    "height": h,
                    "match": match_name if is_match else None
                })
        self.log_debug(f'scan_team {[r["match"] for r in results]}')
        scanner_signals.scan_done.emit(results)
        self.log_info("扫描完成！")