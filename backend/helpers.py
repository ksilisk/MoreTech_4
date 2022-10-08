import db.sqllib as sql
import logging


class Helper:
    def __init__(self) -> None:
        pass

    def add_user(self, data: dict) -> dict:
        logging.info("Helpers add_user func")
        return {"id": sql.add_user(data['name'], data['role'])}
