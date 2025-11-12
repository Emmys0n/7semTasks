from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional

from flask import Flask, abort, jsonify, make_response, request

from .auth import AuthenticationService, InMemoryEmployeeRepository
from .models import Employee, Order, OrderStatus, Role
from .order_manager import OrderManager, StatusTransitionError
from .repositories import (
    InMemoryNotificationGateway,
    InMemoryOrderRepository,
    InMemoryRefundGateway,
    OrderRepository,
)

SESSION_HEADER = "X-Session-Id"


@dataclass
class Session:
    employee: Employee
    sections: tuple
    created_at: datetime


@dataclass
class AppState:
    auth_service: AuthenticationService
    order_manager: OrderManager
    sessions: Dict[str, Session] = field(default_factory=dict)
    current_time: datetime = field(default_factory=datetime.now)

    @property
    def notification_gateway(self) -> InMemoryNotificationGateway:
        return self.order_manager.notification_gateway


def _seed_employees(repo: InMemoryEmployeeRepository) -> None:
    repo.add(Employee(login="BaristaUser1", password="StrongPass1", role=Role.BARISTA))
    repo.add(Employee(login="AdminUser001", password="AdminPass2", role=Role.ADMIN))


def _build_order_manager(repo: Optional[OrderRepository] = None) -> OrderManager:
    repository = repo or InMemoryOrderRepository()
    notification_gateway = InMemoryNotificationGateway()
    refund_gateway = InMemoryRefundGateway()
    return OrderManager(
        repository=repository,
        notification_gateway=notification_gateway,
        refund_gateway=refund_gateway,
    )


def create_app(
    *,
    initial_orders: Optional[Iterable[Order]] = None,
    employee_repo: Optional[InMemoryEmployeeRepository] = None,
) -> Flask:
    app = Flask(__name__)

    employees = employee_repo or InMemoryEmployeeRepository()
    if not employee_repo:
        _seed_employees(employees)

    order_manager = _build_order_manager()
    if initial_orders:
        for order in initial_orders:
            order_manager.register_order(order)

    state = AppState(
        auth_service=AuthenticationService(employee_repo=employees),
        order_manager=order_manager,
    )
    if initial_orders:
        latest_created = max(order.created_at for order in initial_orders)
        state.current_time = latest_created

    app.config["APP_STATE"] = state

    @app.get("/")
    def index():
        return jsonify(
            {
                "message": "Barista order tracking API",
                "endpoints": {
                    "POST /api/login": {
                        "body": {"login": "BaristaUser1", "password": "StrongPass1"}
                    },
                    "GET /api/orders?section=active": {
                        "headers": {"X-Session-Id": "<session_id from login>"}
                    },
                    "POST /api/orders/<order_id>/status": {
                        "body": {"status": "<новый статус>"},
                        "headers": {"X-Session-Id": "<session_id>"},
                    },
                    "POST /api/system/advance-time": {
                        "body": {"minutes": 5},
                        "headers": {"X-Session-Id": "<session_id>"},
                    },
                    "GET /api/notifications": {
                        "headers": {"X-Session-Id": "<session_id>"}
                    },
                },
            }
        )

    @app.get("/ui")
    def ui():
        # Простая HTML-страница с интерфейсом бариста (vanilla JS)
        html = """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Бариста — Панель</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #f6f7f9; color: #222; }
    header { background: #222; color: #fff; padding: 12px 16px; display: flex; align-items: center; gap: 16px; }
    header h1 { margin: 0; font-size: 18px; }
    .container { padding: 16px; }
    .row { display: flex; gap: 16px; align-items: stretch; }
    .card { background: #fff; border: 1px solid #e6e6e6; border-radius: 8px; padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
    .login { display: flex; gap: 8px; align-items: center; }
    input, select, button { padding: 8px 10px; border: 1px solid #ccc; border-radius: 6px; }
    button { background: #2f77f0; color: #fff; border: none; cursor: pointer; }
    button.secondary { background: #666; }
    button.ghost { background: transparent; color: #2f77f0; border: 1px solid #2f77f0; }
    button:disabled { background: #aaa; cursor: not-allowed; }
    .columns { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
    .column h3 { margin: 0 0 8px; }
    .order { border: 1px solid #eee; border-radius: 8px; padding: 8px; margin-bottom: 8px; }
    .muted { color: #666; font-size: 12px; }
    .badge { display: inline-block; padding: 2px 6px; border-radius: 999px; font-size: 12px; margin-left: 4px; }
    .red { background: #ffe5e5; color: #b10808; }
    .orange { background: #fff1df; color: #8a4b00; }
    .green { background: #e6f7e6; color: #176b17; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .section { margin-top: 16px; }
    .orders-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .notif { padding: 8px; border-bottom: 1px solid #eee; }
    .inline { display: inline-flex; gap: 6px; align-items: center; }
  </style>
</head>
<body>
  <header>
    <h1>Панель бариста</h1>
    <div class="login card">
      <input type="text" id="login" value="BaristaUser1" placeholder="Логин (12 символов)">
      <input type="password" id="password" value="StrongPass1" placeholder="Пароль">
      <button id="btnLogin">Войти</button>
      <span id="authStatus" class="muted"></span>
    </div>
    <div class="card inline">
      <button id="btnRefresh">Обновить</button>
      <button id="btnAdvance">+5 минут</button>
      <span class="muted">Текущее время: <span id="curTime">—</span></span>
    </div>
  </header>
  <div class="container">
    <div class="grid">
      <div class="section card">
        <div class="orders-header">
          <h2>Активные заказы</h2>
          <span class="muted">Колонки: «К выполнению», «Выполняется», «Готов к выдаче»</span>
        </div>
        <div class="columns">
          <div class="column">
            <h3>К выполнению</h3>
            <div id="col_todo"></div>
          </div>
          <div class="column">
            <h3>Выполняется</h3>
            <div id="col_inprogress"></div>
          </div>
          <div class="column">
            <h3>Готов к выдаче</h3>
            <div id="col_ready"></div>
          </div>
        </div>
      </div>
      <div class="section card">
        <div class="orders-header">
          <h2>Завершённые заказы</h2>
          <span class="muted">«Получен», «Не получен», «Отменен»</span>
        </div>
        <div class="columns">
          <div class="column">
            <h3>Заказ получен</h3>
            <div id="col_received"></div>
          </div>
          <div class="column">
            <h3>Заказ не получен</h3>
            <div id="col_notreceived"></div>
          </div>
          <div class="column">
            <h3>Заказ отменен</h3>
            <div id="col_cancelled"></div>
          </div>
        </div>
        <div class="section">
          <h3>Уведомления</h3>
          <div id="notifications"></div>
        </div>
      </div>
    </div>
  </div>
  <script>
    const API = {
      login: "/api/login",
      orders: (section) => `/api/orders?section=${section}`,
      status: (id) => `/api/orders/${id}/status`,
      advance: "/api/system/advance-time",
      notifications: "/api/notifications",
    };
    const STATUS = {
      TO_DO: "К выполнению",
      IN_PROGRESS: "Выполняется",
      READY_FOR_PICKUP: "Готов к выдаче",
      RECEIVED: "Заказ получен",
      NOT_RECEIVED: "Заказ не получен",
      CANCELLED: "Заказ отменен",
    };
    const ALLOWED = {
      [STATUS.TO_DO]: [STATUS.IN_PROGRESS, STATUS.CANCELLED],
      [STATUS.IN_PROGRESS]: [STATUS.READY_FOR_PICKUP, STATUS.CANCELLED],
      [STATUS.READY_FOR_PICKUP]: [STATUS.RECEIVED, STATUS.NOT_RECEIVED, STATUS.CANCELLED],
    };
    const PRIORITY_COLOR = {
      "Красный": "red",
      "Оранжевый": "orange",
      "Зеленый": "green",
    };
    function saveSession(id){ localStorage.setItem("session_id", id); }
    function getSession(){ return localStorage.getItem("session_id"); }
    function authHeaders(){ const sid = getSession(); return sid ? { "X-Session-Id": sid } : {}; }
    function showAuthStatus(text, ok=false){
      const el = document.getElementById("authStatus");
      el.textContent = text;
      el.style.color = ok ? "#3a7a27" : "#b10808";
    }
    function badge(priority){
      const cls = PRIORITY_COLOR[priority] || "green";
      return `<span class="badge ${cls}">${priority}</span>`;
    }
    async function doLogin(){
      const login = document.getElementById("login").value.trim();
      const password = document.getElementById("password").value.trim();
      const resp = await fetch(API.login, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ login, password })
      });
      const data = await resp.json();
      if(resp.ok){
        saveSession(data.session_id);
        showAuthStatus(`Вход выполнен: ${data.role}`, true);
        await refreshAll();
      }else{
        showAuthStatus(data.detail || "Ошибка авторизации");
      }
    }
    async function fetchOrders(section){
      const resp = await fetch(API.orders(section), { headers: authHeaders() });
      if(!resp.ok) throw new Error("Ошибка получения заказов");
      return await resp.json();
    }
    async function refreshAll(){
      try{
        const active = await fetchOrders("active");
        const completed = await fetchOrders("completed");
        document.getElementById("curTime").textContent = new Date(active.current_time).toLocaleTimeString();
        renderActive(active.orders);
        renderCompleted(completed.orders);
        await refreshNotifications();
      }catch(e){
        console.error(e);
      }
    }
    function renderActive(groups){
      renderColumn("col_todo", groups[STATUS.TO_DO] || []);
      renderColumn("col_inprogress", groups[STATUS.IN_PROGRESS] || []);
      renderColumn("col_ready", groups[STATUS.READY_FOR_PICKUP] || []);
    }
    function renderCompleted(groups){
      renderColumn("col_received", groups[STATUS.RECEIVED] || [], false);
      renderColumn("col_notreceived", groups[STATUS.NOT_RECEIVED] || [], false);
      renderColumn("col_cancelled", groups[STATUS.CANCELLED] || [], false);
    }
    function renderColumn(containerId, orders, allowActions = true){
      const el = document.getElementById(containerId);
      el.innerHTML = "";
      if(!orders.length){ el.innerHTML = '<div class="muted">—</div>'; return; }
      for(const o of orders){
        const actions = actionButtons(o, allowActions);
        const items = (o.items || []).map(i => `${i.name} x${i.quantity}`).join(", ");
        const pBadge = badge(o.priority);
        const html = `
          <div class="order">
            <div><strong>#${o.order_id.slice(0,8)}</strong> — ${o.customer_name} ${pBadge}</div>
            <div class="muted">${items}</div>
            <div class="muted">Создан: ${new Date(o.created_at).toLocaleTimeString()}</div>
            <div class="muted">Статус: ${o.status}</div>
            ${actions}
          </div>
        `;
        el.insertAdjacentHTML("beforeend", html);
      }
    }
    function actionButtons(order, allow=true){
      if(!allow) return "";
      const possible = ALLOWED[order.status] || [];
      if(!possible.length) return "";
      const btns = possible.map(s => {
        return `<button class="ghost" onclick="changeStatus('${order.order_id}','${s}')">${s}</button>`;
      }).join(" ");
      return `<div style="margin-top:8px">${btns}</div>`;
    }
    async function changeStatus(orderId, newStatus){
      const resp = await fetch(API.status(orderId), {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({ status: newStatus })
      });
      if(resp.ok){
        await refreshAll();
      }else{
        const data = await resp.json().catch(()=>({}));
        alert(data.detail || "Ошибка смены статуса");
      }
    }
    async function advance(){
      const resp = await fetch(API.advance, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({ minutes: 5 })
      });
      if(resp.ok){
        await refreshAll();
      }
    }
    async function refreshNotifications(){
      const resp = await fetch(API.notifications, { headers: authHeaders() });
      if(!resp.ok) return;
      const data = await resp.json();
      const box = document.getElementById("notifications");
      box.innerHTML = "";
      if(!data.notifications.length){
        box.innerHTML = '<div class="muted">Пока нет уведомлений</div>';
        return;
      }
      for(const n of data.notifications){
        box.insertAdjacentHTML("beforeend",
          `<div class="notif"><strong>[${n.type}]</strong> #${n.order_id.slice(0,8)} — ${n.message}</div>`
        );
      }
    }
    document.getElementById("btnLogin").addEventListener("click", doLogin);
    document.getElementById("btnRefresh").addEventListener("click", refreshAll);
    document.getElementById("btnAdvance").addEventListener("click", advance);
    window.changeStatus = changeStatus;
    // Авто-логин если сессия уже есть
    (async ()=>{
      if(getSession()){
        showAuthStatus("Сессия найдена", true);
        await refreshAll();
      }
    })();
  </script>
</body>
</html>
        """.strip()
        return html

    def _unauthorized(message: str):
        response = jsonify({"detail": message})
        response.status_code = 401
        return response

    def _bad_request(message: str, status: int = 400):
        response = jsonify({"detail": message})
        response.status_code = status
        return response

    def require_session() -> Session:
        session_id = request.headers.get(SESSION_HEADER)
        if not session_id:
            abort(make_response(_unauthorized("Требуется авторизация")))
        session = state.sessions.get(session_id)
        if not session:
            abort(make_response(_unauthorized("Сессия не найдена или истекла")))
        return session

    @app.post("/api/login")
    def login():
        payload = request.get_json(silent=True) or {}
        login_value = payload.get("login", "")
        password_value = payload.get("password", "")
        result = state.auth_service.authenticate(login_value, password_value)
        if not result.success:
            if result.error_message == "Неправильный формат введенных данных":
                return _bad_request(result.error_message)
            return _unauthorized(result.error_message)

        employee = result.employee
        sections = state.auth_service.sections_for_employee(employee)
        session_id = uuid.uuid4().hex
        state.sessions[session_id] = Session(
            employee=employee,
            sections=sections,
            created_at=datetime.now(),
        )
        return jsonify(
            {
                "session_id": session_id,
                "role": employee.role.value,
                "sections": list(sections),
            }
        )

    @app.get("/api/orders")
    def get_orders():
        session = require_session()
        if "orders" not in session.sections:
            return _unauthorized("Нет доступа к заказам")
        section = request.args.get("section")
        if section not in {"active", "completed"}:
            return _bad_request("Неизвестный раздел заказов")
        grouped = state.order_manager.get_orders_grouped(section=section)
        response_payload = {
            status.value: [_serialize_order(order) for order in orders]
            for status, orders in grouped.items()
        }
        return jsonify(
            {
                "orders": response_payload,
                "current_time": state.current_time.isoformat(),
            }
        )

    @app.post("/api/orders/<order_id>/status")
    def update_order_status(order_id: str):
        session = require_session()
        if "orders" not in session.sections:
            return _unauthorized("Нет доступа к заказам")

        payload = request.get_json(silent=True) or {}
        status_value = payload.get("status")
        if status_value is None:
            return _bad_request("Не передан статус")
        try:
            new_status = OrderStatus(status_value)
        except ValueError:
            return _bad_request("Неизвестный статус")

        try:
            state.order_manager.update_status(
                order_id,
                new_status,
                current_time=state.current_time,
            )
        except KeyError:
            response = jsonify({"detail": "Заказ не найден"})
            response.status_code = 404
            return response
        except StatusTransitionError as exc:
            return _bad_request(str(exc))

        return jsonify({"ok": True})

    @app.post("/api/system/advance-time")
    def advance_time():
        session = require_session()
        if "orders" not in session.sections:
            return _unauthorized("Нет доступа к управлению временем")

        payload = request.get_json(silent=True) or {}
        minutes = payload.get("minutes", 5)
        try:
            minutes = int(minutes)
        except (ValueError, TypeError):
            return _bad_request("Некорректное значение минут")
        if minutes < 0:
            return _bad_request("Время не может откатываться назад")

        state.current_time = state.current_time + timedelta(minutes=minutes)
        state.order_manager.tick(current_time=state.current_time)

        return jsonify({"current_time": state.current_time.isoformat()})

    @app.get("/api/notifications")
    def list_notifications():
        session = require_session()
        if "orders" not in session.sections:
            return _unauthorized("Нет доступа к уведомлениям")
        notifications = [
            {
                "order_id": item["order_id"],
                "message": item["message"],
                "type": item["type"],
            }
            for item in state.notification_gateway.sent_notifications
        ]
        return jsonify({"notifications": notifications})

    return app


def _serialize_order(order: Order) -> Dict[str, object]:
    return {
        "order_id": order.order_id,
        "status": order.status.value,
        "priority": order.priority.value,
        "customer_name": order.customer_name,
        "payment_method": order.payment_method.value,
        "created_at": order.created_at.isoformat(),
        "last_status_change": order.last_status_change_time().isoformat(),
        "items": [
            {
                "name": item.name,
                "quantity": item.quantity,
                "customizations": item.customizations,
            }
            for item in order.items
        ],
    }

