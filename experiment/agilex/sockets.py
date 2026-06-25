"""Minimal ZMQ inference server (compatible with giga_models.sockets.RobotInferenceServer)."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable

import torch
import zmq


class TorchSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        buffer = BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        buffer = BytesIO(data)
        return torch.load(buffer, map_location="cpu", weights_only=False)


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class BaseInferenceServer:
    def __init__(self, host: str = "*", port: int = 5555):
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, EndpointHandler] = {}
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)

    def _kill_server(self) -> dict[str, Any]:
        self.running = False
        return {"status": "ok", "message": "Server shutting down"}

    def _handle_ping(self) -> dict[str, Any]:
        return {"status": "ok", "message": "Server is running"}

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True) -> None:
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def run(self) -> None:
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Server is ready and listening on {addr}")
        while self.running:
            try:
                message = self.socket.recv()
                request = TorchSerializer.from_bytes(message)
                endpoint = request.get("endpoint", "inference")
                if endpoint not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint}")
                handler = self._endpoints[endpoint]
                if handler.requires_input:
                    result = handler.handler(request.get("data", {}))
                else:
                    result = handler.handler()
                self.socket.send(TorchSerializer.to_bytes(result))
            except Exception as e:
                print(f"Error in server: {e}")
                import traceback

                print(traceback.format_exc())
                self.socket.send(b"ERROR")


class RobotInferenceServer(BaseInferenceServer):
    def __init__(self, model: Any, host: str = "*", port: int = 5555):
        super().__init__(host, port)
        self.register_endpoint("inference", model.inference)
