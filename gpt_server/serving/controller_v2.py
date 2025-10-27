"""
A controller manages distributed workers.
It sends worker addresses to clients.
This version is modified to use SQLModel with SQLite to support
multi-process execution.
"""

import argparse
from enum import Enum, auto
import json
import os
import time
from typing import List, Optional
import threading
import random

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import requests
import uvicorn

# Import SQLModel components
from sqlmodel import Field, SQLModel, create_engine, Session, JSON, Column, select

from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
from loguru import logger

CONTROLLER_HEART_BEAT_EXPIRATION = 30
FASTCHAT_WORKER_API_TIMEOUT = 100

WORKER_API_TIMEOUT = 100


class DispatchMethod(Enum):
    LOTTERY = auto()
    SHORTEST_QUEUE = auto()

    @classmethod
    def from_str(cls, name):
        if name == "lottery":
            return cls.LOTTERY
        elif name == "shortest_queue":
            return cls.SHORTEST_QUEUE
        else:
            raise ValueError(f"Invalid dispatch method")


# NEW: SQLModel definition for a Worker
# This class defines both the database table and the data model
class Worker(SQLModel, table=True):
    # The worker_addr is the worker's address (e.g., "http://localhost:21002")
    worker_addr: str = Field(default=None, primary_key=True)

    # Store the list of model names as a JSON string in the DB
    model_names: List[str] = Field(sa_column=Column(JSON))
    speed: int
    queue_length: int
    check_heart_beat: bool
    last_heart_beat: float  # Use float for time.time()
    multimodal: bool


# NEW: Database setup
# Use a file-based SQLite database. This file will be the shared state.
sqlite_file_name = "controller.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"


engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})


def create_db_and_tables():
    """Creates the database and tables if they don't exist."""
    # 先删后建，确保每次启动都是一张全新的空表
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)


def heart_beat_controller(controller: "Controller"):
    """Periodically removes stale workers from the database."""
    while True:
        time.sleep(CONTROLLER_HEART_BEAT_EXPIRATION)
        controller.remove_stale_workers_by_expiration()


class Controller:
    def __init__(self, dispatch_method: str, db_engine):
        self.engine = db_engine
        self.dispatch_method = DispatchMethod.from_str(dispatch_method)

        self.heart_beat_thread = threading.Thread(
            target=heart_beat_controller, args=(self,)
        )

        self.heart_beat_thread.start()

    def get_session(self):
        """Helper function to get a new database session."""
        return Session(self.engine)

    def register_worker(
        self,
        worker_addr: str,
        check_heart_beat: bool,
        worker_status: dict,
        multimodal: bool,
    ):
        if not worker_status:
            worker_status = self.get_worker_status(worker_addr)
        if not worker_status:
            return False

        with self.get_session() as session:
            # Check if worker already exists

            worker = session.get(Worker, worker_addr)

            if worker:
                # Update existing worker
                logger.info(f"Register (update) an existing worker: {worker_addr}")
                worker.model_names = worker_status["model_names"]
                worker.speed = worker_status["speed"]
                worker.queue_length = worker_status["queue_length"]
                worker.check_heart_beat = check_heart_beat
                worker.last_heart_beat = time.time()
                worker.multimodal = multimodal
            else:
                # Create new worker
                logger.info(f"Register a new worker: {worker_addr}")
                worker = Worker(
                    worker_addr=worker_addr,
                    model_names=worker_status["model_names"],
                    speed=worker_status["speed"],
                    queue_length=worker_status["queue_length"],
                    check_heart_beat=check_heart_beat,
                    last_heart_beat=time.time(),
                    multimodal=multimodal,
                )

            session.add(worker)
            session.commit()
            session.refresh(worker)

        logger.info(f"Register done: {worker_addr}, {worker_status}")
        return True

    def get_worker_status(self, worker_addr: str):
        """(Unchanged) Pings a worker to get its status."""
        try:
            r = requests.post(worker_addr + "/worker_get_status", timeout=5)
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {worker_addr}, {e}")
            return None

        if r.status_code != 200:
            logger.error(f"Get status fails: {worker_addr}, {r}")
            return None

        return r.json()

    def remove_worker(self, worker_addr: str):
        """Removes a worker from the database."""
        with self.get_session() as session:
            worker = session.get(Worker, worker_addr)
            if worker:
                session.delete(worker)
                session.commit()
                logger.info(f"Removed worker: {worker_addr}")
            else:
                logger.warning(
                    f"Attempted to remove non-existent worker: {worker_addr}"
                )

    def refresh_all_workers(self):
        """
        Refreshes status for all workers in the DB.
        Removes any worker that fails the status check.
        """
        with self.get_session() as session:
            statement = select(Worker)
            all_workers = session.exec(statement).all()

        # Iterate over a static list of worker info
        old_info = [
            (w.worker_addr, w.check_heart_beat, w.multimodal) for w in all_workers
        ]

        for w_name, check_hb, multimodal in old_info:
            # register_worker will ping the worker and update its DB entry.
            # If it fails, it returns False.
            if not self.register_worker(w_name, check_hb, None, multimodal):
                logger.info(f"Remove stale worker during refresh: {w_name}")
                # Explicitly remove worker if registration (ping) fails
                self.remove_worker(w_name)

    def list_models(self):
        """Lists all unique models available in the database."""
        model_names = set()
        with self.get_session() as session:
            # Select only the model_names column
            statement = select(Worker.model_names)
            results = session.exec(statement).all()  # List of lists
            for models_list in results:
                model_names.update(models_list)
        return list(model_names)

    def list_multimodal_models(self):
        """Lists models from workers marked as multimodal."""
        model_names = set()
        with self.get_session() as session:
            statement = select(Worker.model_names).where(Worker.multimodal == True)
            results = session.exec(statement).all()
            for models_list in results:
                model_names.update(models_list)
        return list(model_names)

    def list_language_models(self):
        """Lists models from workers not marked as multimodal."""
        model_names = set()
        with self.get_session() as session:
            statement = select(Worker.model_names).where(Worker.multimodal == False)
            results = session.exec(statement).all()
            for models_list in results:
                model_names.update(models_list)
        return list(model_names)

    def get_worker_address(self, model_name: str):

        worker_addrs = []
        with self.get_session() as session:
            # We need all worker info to filter
            statement = select(Worker)
            all_workers = session.exec(statement).all()

            # Filter in Python
            for w in all_workers:
                if model_name in w.model_names:
                    worker_addrs.append(w.worker_addr)

        return ",".join(worker_addrs)

    def receive_heart_beat(self, worker_addr: str, queue_length: int):
        """Updates a worker's heartbeat time and queue length in the DB."""
        with self.get_session() as session:
            worker = session.get(Worker, worker_addr)
            if not worker:
                logger.info(f"Receive unknown heart beat. {worker_addr}")
                return False

            worker.queue_length = queue_length
            worker.last_heart_beat = time.time()
            session.add(worker)
            session.commit()

        return True

    def remove_stale_workers_by_expiration(self):
        """Removes workers from DB that have not sent a heartbeat."""
        expire_time = time.time() - CONTROLLER_HEART_BEAT_EXPIRATION

        with self.get_session() as session:
            # Find all workers that require heartbeats and are expired
            statement = select(Worker).where(
                Worker.check_heart_beat == True, Worker.last_heart_beat < expire_time
            )
            stale_workers = session.exec(statement).all()

            if not stale_workers:
                return

            to_delete_names = [w.worker_addr for w in stale_workers]
            logger.info(f"Removing stale workers: {to_delete_names}")

            for worker in stale_workers:
                session.delete(worker)
            session.commit()

    def handle_no_worker(self, params):
        """(Unchanged) Returns error JSON for no available worker."""
        logger.info(f"no worker: {params['model']}")
        ret = {
            "text": SERVER_ERROR_MSG,
            "error_code": ErrorCode.CONTROLLER_NO_WORKER,
        }
        return json.dumps(ret).encode() + b"\0"

    def handle_worker_timeout(self, worker_address):
        """(Unchanged) Returns error JSON for worker timeout."""
        logger.info(f"worker timeout: {worker_address}")
        ret = {
            "text": SERVER_ERROR_MSG,
            "error_code": ErrorCode.CONTROLLER_WORKER_TIMEOUT,
        }
        return json.dumps(ret).encode() + b"\0"


app = FastAPI()


@app.post("/register_worker")
async def register_worker(request: Request):
    data = await request.json()
    controller.register_worker(
        data["worker_addr"],
        data["check_heart_beat"],
        data.get("worker_status", None),
        data.get("multimodal", False),
    )


@app.post("/refresh_all_workers")
async def refresh_all_workers():
    models = controller.refresh_all_workers()


@app.post("/list_models")
async def list_models():
    models = controller.list_models()
    return {"models": models}


@app.post("/list_multimodal_models")
async def list_multimodal_models():
    models = controller.list_multimodal_models()
    return {"models": models}


@app.post("/list_language_models")
async def list_language_models():
    models = controller.list_language_models()
    return {"models": models}


@app.post("/get_worker_address")
async def get_worker_address(request: Request):
    data = await request.json()
    addr = controller.get_worker_address(data["model"])
    return {"address": addr}


@app.post("/receive_heart_beat")
async def receive_heart_beat(request: Request):
    data = await request.json()
    exist = controller.receive_heart_beat(data["worker_addr"], data["queue_length"])
    return {"exist": exist}


# delete
@app.get("/test_connection")
async def worker_api_get_status(request: Request):
    return "success"


def create_controller(db_engine_to_use):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=51001)
    parser.add_argument(
        "--dispatch-method",
        type=str,
        choices=["lottery", "shortest_queue"],
        default="shortest_queue",
    )
    parser.add_argument(
        "--ssl",
        action="store_true",
        required=False,
        default=False,
        help="Enable SSL. Requires OS Environment variables 'SSL_KEYFILE' and 'SSL_CERTFILE'.",
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # Pass the shared DB engine to the controller instance
    controller_instance = Controller(args.dispatch_method, db_engine_to_use)
    return args, controller_instance


if __name__ == "__main__":
    # 1. Create the database and tables first
    # This is idempotent and safe to run every time.
    create_db_and_tables()

    # 2. Create the controller instance, passing the shared engine
    # This `controller` is the global object used by the API routes
    args, controller = create_controller(engine)

    # 3. Run the FastAPI app
    # If you run this with multiple workers (e.g., `uvicorn ... --workers 4`),
    # each worker process will have its own `controller` object,
    # but all of them will share the *same* `engine` pointing to the
    # same SQLite DB file, achieving shared state.
    if args.ssl:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            ssl_keyfile=os.environ["SSL_KEYFILE"],
            ssl_certfile=os.environ["SSL_CERTFILE"],
        )
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
