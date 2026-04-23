from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
from ok import Logger, communicate, og
from PySide6.QtCore import QObject

logger = Logger.get_logger(__name__)


class Globals(QObject):
    def __init__(self, exit_event):
        super().__init__()
        self._thread_pool_executor_max_workers = 0
        self.thread_pool_executor = None
        self.thread_pool_exit_event = Event()
        exit_event.bind_stop(self)
        self._ocr_init = False
        self.bg_ocr_lib = None
        communicate.starting_emulator.connect(self.ocr_init)

    def ocr_init(self, done, error, seconds_left):
        if done and error == "" and seconds_left == 0 and not self._ocr_init:
            import threading

            t = threading.Thread(
                target=lambda: og.executor.get_all_tasks()[0].ocr(0, 0, 0.01, 0.01)
            )
            t.daemon = True
            t.start()
            self.init_bg_ocr()
            t2 = threading.Thread(
                target=lambda: og.executor._ocr_lib["bg_onnx_ocr"].ocr(
                    np.zeros((50, 50, 3), dtype=np.uint8)
                )
            )
            t2.daemon = True
            t2.start()
            self._ocr_init = True

    def init_bg_ocr(self):
        from onnxocr.onnx_paddleocr import ONNXPaddleOcr

        config_params = og.executor.config.get("ocr").get("default").get("params")
        logger.info(f"init bg onnxocr {config_params}")
        og.executor._ocr_lib["bg_onnx_ocr"] = ONNXPaddleOcr(
            use_angle_cls=False,
            logger=logger,
            use_npu=config_params.get("use_npu", True),
            use_openvino=config_params.get("use_openvino", False),
        )

    def stop(self):
        self.shutdown_thread_pool_executor()

    def get_thread_pool_executor(self, max_workers=6):
        """
        获取全局执行器。
        如果请求的 max_workers 大于当前值，将安全地重建线程池。
        """
        if (
            self.thread_pool_executor is not None
            and max_workers > self._thread_pool_executor_max_workers
        ):
            logger.info(
                "thread pool max_workers not enough, reset max_workers" \
                f" {self._thread_pool_executor_max_workers} -> {max_workers}"
            )
            self.shutdown_thread_pool_executor()

        if self.thread_pool_executor is None:
            logger.info(f"create thread pool executor, max_workers: {max_workers}")
            self.thread_pool_exit_event.clear()
            self.thread_pool_executor = ThreadPoolExecutor(max_workers=max_workers)
            self._thread_pool_executor_max_workers = max_workers

        return self.thread_pool_executor

    def shutdown_thread_pool_executor(self):
        if self.thread_pool_executor is not None:
            logger.info("Shutting down thread pool executor...")
            self.thread_pool_exit_event.set()
            self.thread_pool_executor.shutdown(wait=False, cancel_futures=True)
            self.thread_pool_executor = None
            self._thread_pool_executor_max_workers = 0

    def submit_periodic_task(self, delay, task, *args, **kwargs):
        """
        提交一个循环任务到线程池。
        如果要停止循环，任务函数应返回 False。

        :param task: 要执行的函数
        :param delay: 每次执行后的间隔时间（秒）
        :param args: 位置参数
        :param kwargs: 关键字参数
        """
        executor = self.get_thread_pool_executor()

        def loop_wrapper():
            logger.debug(f"Periodic task {task.__name__} started.")

            while not self.thread_pool_exit_event.is_set():
                should_stop = False
                try:
                    if task(*args, **kwargs) is False:
                        should_stop = True
                except Exception as e:
                    logger.error(f"Error in periodic task {task.__name__}: {e}")

                if should_stop:
                    logger.debug(f"Periodic task {task.__name__} decided to stop.")
                    break

                if self.thread_pool_exit_event.wait(timeout=delay):
                    logger.debug(f"Periodic task {task.__name__} received stop signal.")
                    break

            logger.debug(f"Periodic task {task.__name__} stopped.")

        executor.submit(loop_wrapper)
