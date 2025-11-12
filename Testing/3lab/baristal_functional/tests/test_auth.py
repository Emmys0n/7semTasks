import pytest
from datetime import datetime

from barista_app.auth import AuthenticationService, InMemoryEmployeeRepository
from barista_app.models import Employee, Role


@pytest.fixture
def employee_repo():
    repo = InMemoryEmployeeRepository()
    repo.add(Employee(login="BaristaUser1", password="StrongPass1", role=Role.BARISTA))
    repo.add(Employee(login="AdminUser001", password="AdminPass2", role=Role.ADMIN))
    return repo


@pytest.fixture
def auth_service(employee_repo):
    return AuthenticationService(employee_repo=employee_repo)


def test_login_format_valid(auth_service):
    assert auth_service.validate_login_format("Abcdef123456")


@pytest.mark.parametrize(
    "login",
    ["short1", "toolonglogin123", "invalid!", "русский1234", "UpperCaseOnly"]
)
def test_login_format_invalid(auth_service, login):
    assert not auth_service.validate_login_format(login)


@pytest.mark.parametrize(
    "password",
    ["short", "nouppercase8", "NOLOWERCASE9", "NoDigitsHere", "русскийПароль1"]
)
def test_password_format_invalid(auth_service, password):
    assert not auth_service.validate_password_format(password)


def test_password_format_valid(auth_service):
    assert auth_service.validate_password_format("ValidPass9")


def test_authenticate_success(auth_service):
    result = auth_service.authenticate("BaristaUser1", "StrongPass1")
    assert result.success
    assert result.employee.role == Role.BARISTA


def test_authenticate_wrong_format(auth_service):
    result = auth_service.authenticate("bad", "pwd")
    assert result.error_message == "Неправильный формат введенных данных"


def test_authenticate_wrong_credentials(auth_service):
    result = auth_service.authenticate("BaristaUser1", "WrongPass9")
    assert result.error_message == "Вы ввели неправильный логин или пароль. Повторите попытку."
    assert not result.success

