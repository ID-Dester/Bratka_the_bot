"""Single-process JSON storage, compatible with the original users.json."""

import json
import os
from pathlib import Path


class DataError(ValueError):
    """Existing statistics cannot be read safely."""


def validate(data: object) -> dict[str, dict[str, int]]:
    if not isinstance(data, dict):
        raise DataError("Статистика должна быть JSON-объектом.")
    for guild_id, users in data.items():
        if not isinstance(guild_id, str) or not guild_id.isascii() or not guild_id.isdigit():
            raise DataError("Некорректный ID сервера в статистике.")
        if not isinstance(users, dict):
            raise DataError("Статистика сервера должна быть объектом.")
        for user_id, count in users.items():
            if not isinstance(user_id, str) or not user_id.isascii() or not user_id.isdigit():
                raise DataError("Некорректный ID пользователя в статистике.")
            if type(count) is not int or count < 0:
                raise DataError("Счётчик сообщений должен быть неотрицательным целым числом.")
    return data


class StatsStore:
    def __init__(self, path: Path):
        self.path = path
        self.backup = path.with_suffix(path.suffix + '.bak')

    def load(self) -> dict[str, dict[str, int]]:
        try:
            raw = self.path.read_text(encoding='utf-8')
        except FileNotFoundError:
            if self.backup.exists():
                raise DataError("Основной файл отсутствует, но есть резервная копия. Восстановите её.")
            return {}
        except UnicodeError as error:
            raise DataError(f"Некорректная кодировка {self.path.name}; файл оставлен без изменений.") from error
        try:
            return validate(json.loads(raw))
        except (json.JSONDecodeError, UnicodeError, DataError) as error:
            raise DataError(f"Не удалось прочитать {self.path.name}; файл оставлен без изменений.") from error

    @staticmethod
    def _replace(path: Path, content: str) -> None:
        temporary = path.with_name(path.name + '.tmp')
        try:
            with temporary.open('w', encoding='utf-8', newline='\n') as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def save(self, data: dict[str, dict[str, int]]) -> None:
        content = json.dumps(validate(data), ensure_ascii=False, indent=2) + '\n'
        if self.path.exists():
            # Refuse to overwrite a damaged file, including one changed while running.
            previous = self.load()
            self._replace(self.backup, json.dumps(previous, ensure_ascii=False, indent=2) + '\n')
        self._replace(self.path, content)
