from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple


class OrderStatus(str, Enum):
    TO_DO = "К выполнению"
    IN_PROGRESS = "Выполняется"
    READY_FOR_PICKUP = "Готов к выдаче"
    RECEIVED = "Заказ получен"
    NOT_RECEIVED = "Заказ не получен"
    CANCELLED = "Заказ отменен"

    @property
    def is_active(self) -> bool:
        return self in {
            OrderStatus.TO_DO,
            OrderStatus.IN_PROGRESS,
            OrderStatus.READY_FOR_PICKUP,
        }


class OrderPriority(str, Enum):
    RED = "Красный"
    ORANGE = "Оранжевый"
    GREEN = "Зеленый"

    def sort_index(self) -> int:
        mapping = {
            OrderPriority.RED: 0,
            OrderPriority.ORANGE: 1,
            OrderPriority.GREEN: 2,
        }
        return mapping[self]


class PaymentMethod(str, Enum):
    PAY_ON_PICKUP = "Оплата при получении"
    PAID_ONLINE = "Оплачено онлайн"


class Role(str, Enum):
    BARISTA = "barista"
    ADMIN = "admin"


@dataclass
class OrderItem:
    name: str
    quantity: int
    customizations: Dict[str, str] = field(default_factory=dict)


@dataclass
class Order:
    order_id: str
    created_at: datetime
    status: OrderStatus
    items: List[OrderItem]
    priority: OrderPriority
    payment_method: PaymentMethod
    customer_name: str
    status_history: List[Tuple[OrderStatus, datetime]] = field(default_factory=list)
    priority_sort_override: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.status_history:
            self.status_history.append((self.status, self.created_at))

    def record_status(self, status: OrderStatus, when: datetime) -> None:
        self.status = status
        self.status_history.append((status, when))

    def status_changed_at(self, status: OrderStatus) -> Optional[datetime]:
        for hist_status, timestamp in reversed(self.status_history):
            if hist_status == status:
                return timestamp
        return None

    def last_status_change_time(self) -> datetime:
        return self.status_history[-1][1]


@dataclass
class Employee:
    login: str
    password: str
    role: Role


@dataclass
class AuthenticationResult:
    employee: Optional[Employee] = None
    error_message: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.employee is not None

