from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta
from typing import List, Sequence

from .models import Order, OrderItem, OrderPriority, OrderStatus, PaymentMethod


DRINKS: Sequence[dict] = (
    {"name": "Капучино", "options": {"молоко": ["коровье", "кокосовое", "овсяное"]}},
    {"name": "Раф", "options": {"сироп": ["ваниль", "карамель", "орех"]}},
    {"name": "Эспрессо", "options": {}},
    {"name": "Флэт уайт", "options": {"крепость": ["двойной", "обычный"]}},
    {"name": "Латте", "options": {"сироп": ["орех", "шоколад", "ваниль"]}},
)


def generate_random_orders(count: int, seed: int | None = None) -> List[Order]:
    rng = random.Random(seed)
    now = datetime.now()
    priorities = list(OrderPriority)
    statuses = list(OrderStatus)
    payments = list(PaymentMethod)
    orders: List[Order] = []

    for _ in range(count):
        created_at = now - timedelta(minutes=rng.randint(1, 200))
        priority = rng.choice(priorities)
        payment = rng.choice(payments)
        drink = rng.choice(DRINKS)
        item = OrderItem(
            name=drink["name"],
            quantity=rng.randint(1, 3),
            customizations=_pick_customizations(drink["options"], rng),
        )

        order = Order(
            order_id=str(uuid.uuid4()),
            created_at=created_at,
            status=OrderStatus.TO_DO,
            items=[item],
            priority=priority,
            payment_method=payment,
            customer_name=_fake_name(rng),
        )

        final_status = rng.choice(statuses)
        _simulate_status_path(order, final_status, created_at, rng)
        orders.append(order)
    return orders


def _pick_customizations(options: dict, rng: random.Random) -> dict:
    customizations = {}
    for key, available in options.items():
        customizations[key] = rng.choice(available)
    return customizations


def _fake_name(rng: random.Random) -> str:
    first_names = ["Анна", "Иван", "Мария", "Олег", "Дарья", "Егор", "Екатерина"]
    return rng.choice(first_names)


def _simulate_status_path(
    order: Order,
    final_status: OrderStatus,
    base_time: datetime,
    rng: random.Random,
) -> None:
    timeline = [OrderStatus.TO_DO]

    if final_status == OrderStatus.TO_DO:
        return

    if final_status == OrderStatus.IN_PROGRESS:
        timeline.append(OrderStatus.IN_PROGRESS)
    elif final_status == OrderStatus.READY_FOR_PICKUP:
        timeline.extend([OrderStatus.IN_PROGRESS, OrderStatus.READY_FOR_PICKUP])
    elif final_status == OrderStatus.RECEIVED:
        timeline.extend(
            [
                OrderStatus.IN_PROGRESS,
                OrderStatus.READY_FOR_PICKUP,
                OrderStatus.RECEIVED,
            ]
        )
    elif final_status == OrderStatus.NOT_RECEIVED:
        timeline.extend(
            [
                OrderStatus.IN_PROGRESS,
                OrderStatus.READY_FOR_PICKUP,
                OrderStatus.NOT_RECEIVED,
            ]
        )
    elif final_status == OrderStatus.CANCELLED:
        timeline.append(OrderStatus.CANCELLED)

    elapsed = 0
    order.status_history = []
    for status in timeline:
        moment = base_time + timedelta(minutes=elapsed)
        order.record_status(status, moment)
        elapsed += rng.randint(1, 5)


