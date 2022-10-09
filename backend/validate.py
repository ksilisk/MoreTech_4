import logging
import backend.db.sqllib as sql


class Validate:

    def __init__(self) -> None:
        pass

    def valid_data(self, data: dict) -> bool:
        logging.info("Validate valid_data func")
        if type(data['name']) == str and type(data['role']) == str \
                and data['role'] in ['booker', 'ceo'] and sql.check_name(data['name']):
            return True
        return False

    def valid_id(self, id: int) -> bool:
        return sql.check_id(id)
