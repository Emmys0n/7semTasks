from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional

from .models import Order, OrderPriority, OrderStatus, PaymentMethod
from .repositories import (
    InMemoryNotificationGateway,
    InMemoryOrderRepository,
    InMemoryRefundGateway,
    OrderRepository,
    ScheduledNotification,
)


class StatusTransitionError(ValueError):
    pass


ALLOWED_TRANSITIONS: Dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.TO_DO: {OrderStatus.IN_PROGRESS, OrderStatus.CANCELLED},
    OrderStatus.IN_PROGRESS: {OrderStatus.READY_FOR_PICKUP, OrderStatus.CANCELLED},
    OrderStatus.READY_FOR_PICKUP: {
        OrderStatus.RECEIVED,
        OrderStatus.NOT_RECEIVED,
        OrderStatus.CANCELLED,
    },
    OrderStatus.RECEIVED: set(),
    OrderStatus.NOT_RECEIVED: set(),
    OrderStatus.CANCELLED: set(),
}


ACTIVE_STATUSES = [
    OrderStatus.TO_DO,
    OrderStatus.IN_PROGRESS,
    OrderStatus.READY_FOR_PICKUP,
]

COMPLETED_STATUSES = [
    OrderStatus.RECEIVED,
    OrderStatus.NOT_RECEIVED,
    OrderStatus.CANCELLED,
]


@dataclass
class OrderManager:
    repository: OrderRepository
    notification_gateway: InMemoryNotificationGateway
    refund_gateway: InMemoryRefundGateway
    _scheduled_notifications: List[ScheduledNotification] = field(default_factory=list)

    def register_order(self, order: Order) -> None:
        self.repository.add(order)

    def get_orders_grouped(self, *, section: str) -> Dict[OrderStatus, List[Order]]:
        if section not in {"active", "completed"}:
            raise ValueError("Unknown section")
        buckets: Dict[OrderStatus, List[Order]] = {
            status: [] for status in (ACTIVE_STATUSES if section == "active" else COMPLETED_STATUSES)
        }
        for order in self.repository.all_orders():
            if section == "active" and order.status in ACTIVE_STATUSES:
                buckets[order.status].append(order)
            if section == "completed" and order.status in COMPLETED_STATUSES:
                buckets[order.status].append(order)

        for status, orders in buckets.items():
            if status == OrderStatus.TO_DO:
                orders.sort(
                    key=lambda o: (
                        o.priority_sort_override if o.priority_sort_override is not None else o.priority.sort_index(),
                        o.created_at,
                    )
                )
            elif status in {OrderStatus.IN_PROGRESS, OrderStatus.READY_FOR_PICKUP}:
                orders.sort(key=lambda o: o.status_changed_at(status) or o.created_at)
            else:
                orders.sort(key=lambda o: o.status_changed_at(status) or o.created_at)
        return buckets

    def update_status(
        self,
        order_id: str,
        new_status: OrderStatus,
        current_time: Optional[datetime] = None,
    ) -> None:
        order = self.repository.get(order_id)
        current_time = current_time or datetime.now()
        self._validate_transition(order.status, new_status)
        self._set_status(order, new_status, current_time, validate=False)

    def _set_status(
        self,
        order: Order,
        new_status: OrderStatus,
        when: datetime,
        *,
        validate: bool = True,
    ) -> None:
        if validate:
            self._validate_transition(order.status, new_status)
        order.record_status(new_status, when)
        self.repository.update(order)
        self._handle_post_status_change(order, new_status, when)

    def _handle_post_status_change(
        self, order: Order, new_status: OrderStatus, when: datetime
    ) -> None:
        # Clear scheduled notifications for non-ready statuses
        if new_status != OrderStatus.READY_FOR_PICKUP:
            self._scheduled_notifications = [
                n for n in self._scheduled_notifications if n.order_id != order.order_id
            ]

        if new_status == OrderStatus.READY_FOR_PICKUP:
            self._notify_ready(order)
            first_reminder_time = when + timedelta(minutes=5)
            self._scheduled_notifications.append(
                ScheduledNotification(
                    order_id=order.order_id,
                    send_at=first_reminder_time,
                    notification_type="reminder",
                )
            )
        elif new_status == OrderStatus.CANCELLED:
            if order.payment_method == PaymentMethod.PAID_ONLINE:
                self.refund_gateway.process_refund(order.order_id)
            self.notification_gateway.send(
                order.order_id,
                f"Заказ {order.order_id} отменен.",
                notification_type="status_change",
            )
        elif new_status == OrderStatus.RECEIVED:
            self.notification_gateway.send(
                order.order_id,
                f"Заказ {order.order_id} получен клиентом.",
                notification_type="status_change",
            )

    def _notify_ready(self, order: Order) -> None:
        self.notification_gateway.send(
            order.order_id,
            f"Заказ {order.order_id} готов к выдаче для клиента {order.customer_name}.",
            notification_type="ready",
        )

    def _validate_transition(self, current: OrderStatus, new: OrderStatus) -> None:
        allowed = ALLOWED_TRANSITIONS[current]
        if new not in allowed:
            raise StatusTransitionError(
                f"Cannot move order from {current.value} to {new.value}"
            )

    def tick(self, current_time: Optional[datetime] = None) -> None:
        current_time = current_time or datetime.now()
        self._process_notifications(current_time)
        self._process_ready_timeouts(current_time)

    def _process_notifications(self, current_time: datetime) -> None:
        pending: List[ScheduledNotification] = []
        for notification in self._scheduled_notifications:
            if notification.send_at <= current_time:
                order = self.repository.get(notification.order_id)
                if order.status == OrderStatus.READY_FOR_PICKUP:
                    self.notification_gateway.send(
                        order.order_id,
                        f"Заказ {order.order_id} готов к выдаче. Клиент еще не забрал заказ.",
                        notification_type=notification.notification_type,
                    )
                    reminder_time = notification.send_at + timedelta(minutes=3)
                    pending.append(
                        ScheduledNotification(
                            order_id=order.order_id,
                            send_at=reminder_time,
                            notification_type=notification.notification_type,
                        )
                    )
            else:
                pending.append(notification)
        self._scheduled_notifications = pending

    def _process_ready_timeouts(self, current_time: datetime) -> None:
        for order in list(self.repository.all_orders()):
            if order.status == OrderStatus.READY_FOR_PICKUP:
                ready_since = order.status_changed_at(OrderStatus.READY_FOR_PICKUP)
                if ready_since and current_time - ready_since >= timedelta(minutes=90):
                    self._set_status(order, OrderStatus.NOT_RECEIVED, current_time, validate=False)

