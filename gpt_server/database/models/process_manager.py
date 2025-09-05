"""暂时没有使用此代码"""

from typing import List, Dict, Optional, Any
from multiprocessing import Process
from sqlmodel import SQLModel, Field, create_engine, Session, select
from datetime import datetime
import json
from uuid import uuid4


# 数据库模型
class ProcessRecord(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True, description="主键ID")
    pid: int | None = Field(default=None, description="进程ID")
    args: str = Field(default="", description="进程参数")
    status: str = Field(
        default="created", description="进程状态"
    )  # created, started, stopped
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    started_at: Optional[datetime] = Field(default=None, description="启动时间")
    stopped_at: Optional[datetime] = Field(default=None, description="停止时间")


class ProcessManager:
    def __init__(self, write_db: bool = False, db_url: str = "sqlite:///processes.db"):
        """进程管理类

        Parameters
        ----------
        write_db : bool, optional
            是否将进程信息写入到数据库, by default False
        db_url : str, optional
            数据库的连接 url, by default "sqlite:///processes.db"
        """
        self.processes: List[Dict[Process, dict]] | None = []
        self.write_db = write_db
        if self.write_db:
            self.engine = create_engine(db_url)
            # 创建表
            SQLModel.metadata.create_all(self.engine)

    def add_process(
        self,
        target,
        args=(),
    ):
        p = Process(target=target, args=args)
        process_id = uuid4().int & ((1 << 64) - 1)
        self.processes.append({p: {"args": args, "process_id": process_id}})
        if self.write_db:
            # 记录到数据库
            with Session(self.engine) as session:

                process_record = ProcessRecord(
                    id=process_id,
                    pid=None,
                    args=json.dumps(args, ensure_ascii=False),
                    status="created",
                )
                session.add(process_record)
                session.commit()
                session.refresh(process_record)

    def start_all(self):
        for process in self.processes:
            for _process, process_info in process.items():
                _process.start()
                process_info["pid"] = _process.pid
                if self.write_db:
                    process_id = process_info["process_id"]
                    # 更新数据库记录
                    with Session(self.engine) as session:
                        # 根据PID查找记录（这里简化处理，实际可能需要更好的标识）
                        statement = select(ProcessRecord).where(
                            ProcessRecord.id == process_id
                        )
                        result = session.exec(statement)
                        process_record = result.first()
                        if process_record:
                            process_record.pid = _process.pid
                            process_record.status = "started"
                            process_record.started_at = datetime.now()
                            session.add(process_record)
                            session.commit()
                            session.refresh(process_record)

    def join_all(self):
        for process in self.processes:
            for _process, process_info in process.items():
                _process.join()
                if self.write_db:
                    process_id = process_info["process_id"]
                    # 更新数据库记录为完成状态
                    with Session(self.engine) as session:
                        statement = select(ProcessRecord).where(
                            ProcessRecord.id == process_id
                        )
                        results = session.exec(statement)
                        record = results.first()
                        if record:
                            record.status = "finished"
                            record.finished_at = datetime.now()
                            session.add(record)
                            session.commit()
