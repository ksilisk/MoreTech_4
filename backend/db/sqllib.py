import sqlite3
import logging

con = sqlite3.connect('db/database.db', check_same_thread=False)
cur = con.cursor()


def add_user(name: str, role: str) -> int:
    logging.info('SQL - add_user func for data %s %s' % (name, role))
    cur.execute("INSERT INTO users (name, role) VALUES (?, ?)", (name, role))
    con.commit()
    return get_id_by_name(name)


def get_id_by_name(name: str) -> int:
    logging.info('SQL - get_id_by_name func for data %s' % (name))
    return int(cur.execute("SELECT id FROM users WHERE name = ?", (name,)).fetchone()[0])


def check_name(name: str) -> bool:
    res = cur.execute("SELECT id FROM users WHERE name = ?", (name,)).fetchone()
    return True if not res else False


def check_id(user_id: int) -> bool:
    res = cur.execute("SELECT name FROM users WHERE id = ?", (user_id,)).fetchone()
    return True if res else False


def add_news(user_id: int, title: str) -> None:
    cur.execute("INSERT INTO news (user_id, title) VALUES (?, ?)", (user_id, title))
    con.commit()


def get_role(user_id: int) -> str:
    return cur.execute("SELECT role FROM users WHERE id = ?", (user_id,)).fetchone()[0]
