"""兼容入口：历史启动文件。实际应用与实现见 api.main（uvicorn api.main:app）。"""

from api.main import app

__all__ = ["app"]
