from datetime import datetime, timedelta

import pytest

from barista_app.models import (
    Order,
    OrderItem,
    OrderPriority,
    OrderStatus,
    PaymentMethod,
)
from barista_app.webapp import create_app


def make_order(
    order_id: str,
    status: OrderStatus,
    created_at: datetime,
    priority: OrderPriority = OrderPriority.ORANGE,
    payment: PaymentMethod = PaymentMethod.PAY_ON_PICKUP,
):
    order = Order(
        order_id=order_id,
        created_at=created_at,
        status=status,
        items=[OrderItem(name="Латте", quantity=1, customizations={"молоко": "овсяное"})],
        priority=priority,
        payment_method=payment,
        customer_name="Анна",
    )
    return order


@pytest.fixture
def app():
    now = datetime(2024, 1, 1, 9, 0)
    orders = [
        make_order("100", OrderStatus.TO_DO, now, OrderPriority.RED),
        make_order("101", OrderStatus.TO_DO, now + timedelta(minutes=1), OrderPriority.GREEN),
        make_order("102", OrderStatus.IN_PROGRESS, now + timedelta(minutes=2)),
    ]
    application = create_app(initial_orders=orders)
    application.config.update(TESTING=True)
    return application


@pytest.fixture
def client(app):
    with app.test_client() as client:
        yield client


def login(client, login="BaristaUser1", password="StrongPass1"):
    response = client.post(
        "/api/login",
        json={"login": login, "password": password},
    )
    return response


def auth_headers(session_id: str):
    return {"X-Session-Id": session_id}


def test_login_success(client):
    response = login(client)
    data = response.get_json()
    assert response.status_code == 200
    assert data["sections"] == ["orders"]
    assert data["role"] == "barista"
    assert "session_id" in data


def test_login_invalid_format(client):
    response = login(client, login="bad", password="pwd")
    assert response.status_code == 400
    data = response.get_json()
    assert data["detail"] == "Неправильный формат введенных данных"


def test_get_orders_requires_auth(client):
    response = client.get("/api/orders?section=active")
    assert response.status_code == 401


def test_get_active_orders_sorted(client):
    session_id = login(client).get_json()["session_id"]
    response = client.get("/api/orders?section=active", headers=auth_headers(session_id))
    assert response.status_code == 200
    data = response.get_json()
    to_do = data["orders"][OrderStatus.TO_DO.value]
    assert [order["order_id"] for order in to_do] == ["100", "101"]


def test_status_transitions_via_api(client):
    session_id = login(client).get_json()["session_id"]
    headers = auth_headers(session_id)

    response = client.post(
        "/api/orders/100/status",
        json={"status": OrderStatus.IN_PROGRESS.value},
        headers=headers,
    )
    assert response.status_code == 200

    response = client.post(
        "/api/orders/100/status",
        json={"status": OrderStatus.TO_DO.value},
        headers=headers,
    )
    assert response.status_code == 400


def test_ready_notifications_and_time_advance(client):
    session_id = login(client).get_json()["session_id"]
    headers = auth_headers(session_id)

    response = client.post(
        "/api/orders/100/status",
        json={"status": OrderStatus.IN_PROGRESS.value},
        headers=headers,
    )
    assert response.status_code == 200

    response = client.post(
        "/api/orders/100/status",
        json={"status": OrderStatus.READY_FOR_PICKUP.value},
        headers=headers,
    )
    assert response.status_code == 200

    tick_response = client.post(
        "/api/system/advance-time",
        json={"minutes": 6},
        headers=headers,
    )
    assert tick_response.status_code == 200

    notifications_response = client.get("/api/notifications", headers=headers)
    notifications = notifications_response.get_json()["notifications"]
    assert any(note["type"] == "reminder" for note in notifications)

