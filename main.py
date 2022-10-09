"""
Входная точка в сервис, весь запуск происходит отсюда
"""

from backend.app import app


def main():
    app.run()


if __name__ == '__main__':
    main()
