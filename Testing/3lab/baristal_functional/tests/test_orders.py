from datetime import datetime, timedelta

import pytest

from barista_app.models import (
    Order,
    OrderItem,
    OrderPriority,
    OrderStatus,
    PaymentMethod,
    Role,
    Employee,
)
from barista_app.order_manager import OrderManager
from barista_app.repositories import (
    InMemoryOrderRepository,
    InMemoryNotificationGateway,
    InMemoryRefundGateway,
)
from barista_app.random_orders import generate_random_orders


@pytest.fixture
def order_manager():
    return OrderManager(
        repository=InMemoryOrderRepository(),
        notification_gateway=InMemoryNotificationGateway(),
        refund_gateway=InMemoryRefundGateway(),
    )


def create_order(
    order_id: str,
    status: OrderStatus,
    created_at: datetime,
    priority: OrderPriority = OrderPriority.ORANGE,
    payment: PaymentMethod = PaymentMethod.PAY_ON_PICKUP,
) -> Order:
    return Order(
        order_id=order_id,
        created_at=created_at,
        status=status,
        items=[
            OrderItem(name="Flat White", quantity=1, customizations={"milk": "oat"})
        ],
        priority=priority,
        payment_method=payment,
        customer_name="Ivan",
    )


def test_register_and_group_orders(order_manager):
    now = datetime(2024, 1, 1, 9, 0)
    order1 = create_order("1", OrderStatus.TO_DO, now)
    order2 = create_order("2", OrderStatus.IN_PROGRESS, now + timedelta(minutes=1))
    order3 = create_order("3", OrderStatus.RECEIVED, now + timedelta(minutes=2))

    order_manager.register_order(order1)
    order_manager.register_order(order2)
    order_manager.register_order(order3)

    active = order_manager.get_orders_grouped(section="active")
    completed = order_manager.get_orders_grouped(section="completed")

    assert list(active.keys()) == [
        OrderStatus.TO_DO,
        OrderStatus.IN_PROGRESS,
        OrderStatus.READY_FOR_PICKUP,
    ]
    assert list(completed.keys()) == [
        OrderStatus.RECEIVED,
        OrderStatus.NOT_RECEIVED,
        OrderStatus.CANCELLED,
    ]
    assert active[OrderStatus.TO_DO][0].order_id == "1"
    assert completed[OrderStatus.RECEIVED][0].order_id == "3"


def test_to_do_orders_sorted_by_priority_and_time(order_manager):
    now = datetime(2024, 1, 1, 9, 0)
    orders = [
        create_order("1", OrderStatus.TO_DO, now, OrderPriority.GREEN),
        create_order("2", OrderStatus.TO_DO, now + timedelta(minutes=1), OrderPriority.RED),
        create_order("3", OrderStatus.TO_DO, now + timedelta(minutes=2), OrderPriority.ORANGE),
        create_order("4", OrderStatus.TO_DO, now + timedelta(minutes=3), OrderPriority.RED),
    ]
    for order in orders:
        order_manager.register_order(order)

    grouped = order_manager.get_orders_grouped(section="active")
    to_do_ids = [order.order_id for order in grouped[OrderStatus.TO_DO]]

    assert to_do_ids == ["2", "4", "3", "1"]


def test_in_progress_sorted_by_status_time(order_manager):
    now = datetime(2024, 1, 1, 9, 0)
    order1 = create_order("1", OrderStatus.TO_DO, now)
    order2 = create_order("2", OrderStatus.TO_DO, now)
    order3 = create_order("3", OrderStatus.TO_DO, now)
    order_manager.register_order(order1)
    order_manager.register_order(order2)
    order_manager.register_order(order3)

    order_manager.update_status("1", OrderStatus.IN_PROGRESS, now + timedelta(minutes=1))
    order_manager.update_status("2", OrderStatus.IN_PROGRESS, now + timedelta(minutes=2))
    order_manager.update_status("3", OrderStatus.IN_PROGRESS, now + timedelta(minutes=3))

    grouped = order_manager.get_orders_grouped(section="active")
    in_progress_ids = [order.order_id for order in grouped[OrderStatus.IN_PROGRESS]]

    assert in_progress_ids == ["1", "2", "3"]


def test_status_transition_rules(order_manager):
    now = datetime(2024, 1, 1, 9, 0)
    order = create_order("1", OrderStatus.TO_DO, now)
    order_manager.register_order(order)

    order_manager.update_status("1", OrderStatus.IN_PROGRESS, now + timedelta(minutes=1))

    with pytest.raises(ValueError):
        order_manager.update_status("1", OrderStatus.TO_DO, now + timedelta(minutes=2))

    order_manager.update_status("1", OrderStatus.READY_FOR_PICKUP, now + timedelta(minutes=3))
    order_manager.update_status("1", OrderStatus.CANCELLED, now + timedelta(minutes=4))
    assert order_manager.repository.get("1").status == OrderStatus.CANCELLED


def test_ready_status_notifications(order_manager):
    now = datetime(2024, 1, 1, 9, 0)
    order = create_order("1", OrderStatus.TO_DO, now)
    order_manager.register_order(order)

    order_manager.update_status("1", OrderStatus.IN_PROGRESS, now + timedelta(minutes=3))
    order_manager.update_status("1", OrderStatus.READY_FOR_PICKUP, now + timedelta(minutes=5))

    notifications = order_manager.notification_gateway.sent_notifications
    assert notifications[-1]["message"].startswith("Заказ 1 готов к выдаче")

    order_manager.tick(now + timedelta(minutes=10))
    notifications = order_manager.notification_gateway.sent_notifications
    assert len([n for n in notifications if n["type"] == "reminder"]) >= 1


def test_ready_status_auto_not_received(order_manager):
    now = datetime(2024, 1, 1, 9, 0)
    order = create_order("1", OrderStatus.TO_DO, now)
    order_manager.register_order(order)
    ready_time = now + timedelta(minutes=5)
    order_manager.update_status("1", OrderStatus.IN_PROGRESS, now + timedelta(minutes=2))
    order_manager.update_status("1", OrderStatus.READY_FOR_PICKUP, ready_time)

    order_manager.tick(ready_time + timedelta(minutes=91))

    assert order_manager.repository.get("1").status == OrderStatus.NOT_RECEIVED


def test_cancelled_online_orders_trigger_refund(order_manager):
    now = datetime(2024, 1, 1, 9, 0)
    order = create_order(
        "1",
        OrderStatus.TO_DO,
        now,
        payment=PaymentMethod.PAID_ONLINE,
    )
    order_manager.register_order(order)
    order_manager.update_status("1", OrderStatus.CANCELLED, now + timedelta(minutes=1))

    assert order_manager.refund_gateway.refunds == ["1"]


def test_completed_orders_move_sections(order_manager):
    now = datetime(2024, 1, 1, 9, 0)
    order = create_order("1", OrderStatus.TO_DO, now)
    order_manager.register_order(order)
    order_manager.update_status("1", OrderStatus.IN_PROGRESS, now + timedelta(minutes=1))
    order_manager.update_status("1", OrderStatus.READY_FOR_PICKUP, now + timedelta(minutes=2))
    order_manager.update_status("1", OrderStatus.RECEIVED, now + timedelta(minutes=3))

    grouped = order_manager.get_orders_grouped(section="completed")
    assert grouped[OrderStatus.RECEIVED][0].order_id == "1"


def test_generate_random_orders():
    orders = generate_random_orders(5, seed=42)
    assert len(orders) == 5
    for order in orders:
        assert isinstance(order.order_id, str)
        assert order.items
        assert order.customer_name

