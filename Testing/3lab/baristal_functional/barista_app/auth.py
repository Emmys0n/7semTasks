from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Protocol

from .models import AuthenticationResult, Employee, Role


class EmployeeRepository(Protocol):
    def add(self, employee: Employee) -> None:
        ...

    def find_by_login(self, login: str) -> Optional[Employee]:
        ...


class InMemoryEmployeeRepository:
    def __init__(self) -> None:
        self._employees: Dict[str, Employee] = {}

    def add(self, employee: Employee) -> None:
        self._employees[employee.login] = employee

    def find_by_login(self, login: str) -> Optional[Employee]:
        return self._employees.get(login)

    def all_employees(self) -> Iterable[Employee]:
        return self._employees.values()


@dataclass
class AccessControl:
    role_sections: Dict[Role, tuple] = None

    def __post_init__(self) -> None:
        if self.role_sections is None:
            self.role_sections = {
                Role.ADMIN: ("orders", "admin"),
                Role.BARISTA: ("orders",),
            }

    def sections_for(self, role: Role) -> tuple:
        return self.role_sections.get(role, tuple())


class AuthenticationService:
    LOGIN_PATTERN = re.compile(r"^[A-Za-z0-9]{12}$")
    PASSWORD_PATTERN = re.compile(
        r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[A-Za-z\d]{8,}$"
    )

    def __init__(self, employee_repo: EmployeeRepository, access_control: Optional[AccessControl] = None) -> None:
        self.employee_repo = employee_repo
        self.access_control = access_control or AccessControl()

    def validate_login_format(self, login: str) -> bool:
        return bool(self.LOGIN_PATTERN.match(login))

    def validate_password_format(self, password: str) -> bool:
        return bool(self.PASSWORD_PATTERN.match(password))

    def authenticate(self, login: str, password: str) -> AuthenticationResult:
        if not (self.validate_login_format(login) and self.validate_password_format(password)):
            return AuthenticationResult(error_message="Неправильный формат введенных данных")

        employee = self.employee_repo.find_by_login(login)
        if employee and employee.password == password:
            return AuthenticationResult(employee=employee)
        return AuthenticationResult(error_message="Вы ввели неправильный логин или пароль. Повторите попытку.")

    def sections_for_employee(self, employee: Employee) -> tuple:
        return self.access_control.sections_for(employee.role)

