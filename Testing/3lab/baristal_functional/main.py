from barista_app.random_orders import generate_random_orders
from barista_app.webapp import create_app


def main() -> None:
    initial_orders = generate_random_orders(15)
    app = create_app(initial_orders=initial_orders)
    app.run(host="0.0.0.0", port=8000, debug=False)


if __name__ == "__main__":
    main()
