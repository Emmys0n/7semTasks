from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional

from .models import Order


class OrderRepository:
    def add(self, order: Order) -> None:
        raise NotImplementedError

    def get(self, order_id: str) -> Order:
        raise NotImplementedError

    def update(self, order: Order) -> None:
        raise NotImplementedError

    def all_orders(self) -> Iterable[Order]:
        raise NotImplementedError


class InMemoryOrderRepository(OrderRepository):
    def __init__(self) -> None:
        self._storage: Dict[str, Order] = {}

    def add(self, order: Order) -> None:
        if order.order_id in self._storage:
            raise ValueError(f"Order {order.order_id} already exists")
        self._storage[order.order_id] = order

    def get(self, order_id: str) -> Order:
        if order_id not in self._storage:
            raise KeyError(order_id)
        return self._storage[order_id]

    def update(self, order: Order) -> None:
        if order.order_id not in self._storage:
            raise KeyError(order.order_id)
        self._storage[order.order_id] = order

    def all_orders(self) -> Iterable[Order]:
        return list(self._storage.values())


@dataclass
class ScheduledNotification:
    order_id: str
    send_at: datetime
    notification_type: str


@dataclass
class InMemoryNotificationGateway:
    sent_notifications: List[Dict[str, str]] = field(default_factory=list)

    def send(self, order_id: str, message: str, *, notification_type: str) -> None:
        self.sent_notifications.append(
            {"order_id": order_id, "message": message, "type": notification_type}
        )


@dataclass
class InMemoryRefundGateway:
    refunds: List[str] = field(default_factory=list)

    def process_refund(self, order_id: str) -> None:
        self.refunds.append(order_id)

