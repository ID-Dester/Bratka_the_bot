"""Братка: slash-команды и статистика Discord. Made by ID_Dester with love."""

import asyncio
import contextlib
import json
import logging
import random
import signal
from dataclasses import dataclass, field
from pathlib import Path

import discord
from discord import app_commands
from groq import APIError, AsyncGroq, RateLimitError

from storage import DataError, StatsStore

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    token: str = field(repr=False)
    groq_key: str = field(default='', repr=False)
    groq_model: str = 'llama-3.1-8b-instant'


def load_config(path: Path) -> Config:
    try:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as error:
        raise ValueError('Не удалось прочитать config.json. Скопируйте config.example.json и заполните token.') from error
    if not isinstance(data, dict):
        raise ValueError('config.json должен содержать JSON-объект.')
    token = data.get('token')
    key = data.get('groq_key', '')
    model = data.get('groq_model', 'llama-3.1-8b-instant')
    if not isinstance(token, str) or not token.strip():
        raise ValueError('В config.json нужен непустой token.')
    if not isinstance(key, str) or not isinstance(model, str) or not model.strip():
        raise ValueError('groq_key и groq_model должны быть строками; модель не может быть пустой.')
    return Config(token.strip(), key.strip(), model.strip())


def split_message(content: str, limit: int = 1900) -> list[str]:
    """Conservatively count UTF-16 units, so emoji also fit Discord's limit."""
    chunks, current, size = [], [], 0
    for character in content:
        width = 2 if ord(character) > 0xFFFF else 1
        if size + width > limit:
            chunks.append(''.join(current))
            current, size = [], 0
        current.append(character)
        size += width
    if current:
        chunks.append(''.join(current))
    return chunks


async def reply(interaction: discord.Interaction, content: str, *, ephemeral: bool = False) -> None:
    for chunk in split_message(content):
        kwargs = {'ephemeral': ephemeral, 'allowed_mentions': discord.AllowedMentions.none()}
        if interaction.response.is_done():
            await interaction.followup.send(chunk, **kwargs)
        else:
            await interaction.response.send_message(chunk, **kwargs)


class Bratka(discord.Client):
    def __init__(self, config: Config, store: StatsStore):
        super().__init__(intents=discord.Intents(guilds=True, guild_messages=True),
                         allowed_mentions=discord.AllowedMentions.none())
        self.config = config
        self.store = store
        self.stats = store.load()
        self.tree = app_commands.CommandTree(self)
        self.groq = None
        self.ask_slots = asyncio.Semaphore(2)
        self.save_lock = asyncio.Lock()
        self.stop_saving = asyncio.Event()
        self.autosave_task = None
        self.closing = False
        self.register_commands()

    async def setup_hook(self) -> None:
        if self.config.groq_key:
            self.groq = AsyncGroq(api_key=self.config.groq_key, timeout=30.0, max_retries=0)
        await self.tree.sync()
        self.autosave_task = asyncio.create_task(self.autosave(), name='statistics-autosave')

    async def on_ready(self) -> None:
        LOG.info('Братка подключён: %s', self.user)

    async def on_message(self, message: discord.Message) -> None:
        if self.closing or message.author.bot or message.guild is None:
            return
        users = self.stats.setdefault(str(message.guild.id), {})
        user_id = str(message.author.id)
        users[user_id] = users.get(user_id, 0) + 1

    async def save_stats(self) -> None:
        async with self.save_lock:
            snapshot = {guild: dict(users) for guild, users in self.stats.items()}
            await asyncio.to_thread(self.store.save, snapshot)

    async def autosave(self) -> None:
        while not self.stop_saving.is_set():
            try:
                await asyncio.wait_for(self.stop_saving.wait(), timeout=300)
            except TimeoutError:
                try:
                    await self.save_stats()
                except (OSError, DataError):
                    LOG.exception('Не удалось сохранить статистику; повтор через 5 минут.')

    async def close(self) -> None:
        if self.closing:
            return
        self.closing = True
        self.stop_saving.set()
        try:
            if self.autosave_task:
                await self.autosave_task
            await self.save_stats()
        finally:
            try:
                if self.groq is not None:
                    await self.groq.close()
            finally:
                await super().close()

    def register_commands(self) -> None:
        @self.tree.command(name='ping', description='Проверить, отвечает ли бот')
        async def ping(interaction: discord.Interaction):
            await reply(interaction, 'pong')

        @self.tree.command(name='top', description='10 самых активных участников сервера')
        @app_commands.guild_only()
        async def top(interaction: discord.Interaction):
            await interaction.response.defer()
            users = self.stats.get(str(interaction.guild_id), {})
            if not users:
                await reply(interaction, 'На этом сервере ещё нет статистики.')
                return
            leaders = sorted(users.items(), key=lambda item: item[1], reverse=True)[:10]
            lines = []
            for position, (user_id, count) in enumerate(leaders, 1):
                user = self.get_user(int(user_id))
                if user is None:
                    try:
                        user = await self.fetch_user(int(user_id))
                    except discord.HTTPException:
                        LOG.warning('Не удалось получить пользователя %s', user_id, exc_info=True)
                name = str(user) if user else f'Пользователь {user_id}'
                lines.append(f'{position}. {discord.utils.escape_markdown(name)} — {count} сообщений')
            await reply(interaction, '\n'.join(lines))

        @self.tree.command(name='difficult_choice', description='Выбрать вариант из списка через запятую')
        @app_commands.describe(choices='Варианты через запятую (до 900 символов)')
        async def choice(interaction: discord.Interaction, choices: app_commands.Range[str, 1, 900]):
            options = [option.strip() for option in choices.split(',') if option.strip()]
            if not options:
                await reply(interaction, 'Введи хотя бы один вариант.', ephemeral=True)
                return
            winner = discord.utils.escape_markdown(random.choice(options))
            await reply(interaction, f'Мой выбор: **{winner}**\nВариантов: {len(options)}')

        @self.tree.command(name='ask', description='Задать вопрос нейросети через Groq')
        @app_commands.describe(question='Вопрос (до 1000 символов)')
        @app_commands.checks.cooldown(1, 10.0, key=lambda interaction: interaction.user.id)
        async def ask(interaction: discord.Interaction, question: app_commands.Range[str, 1, 1000]):
            await self.ask_question(interaction, question)

        @self.tree.error
        async def command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
            if isinstance(error, app_commands.CommandOnCooldown):
                message = f'Подожди {error.retry_after:.0f} сек. перед следующим вопросом.'
            else:
                LOG.error('Ошибка slash-команды', exc_info=(type(error), error, error.__traceback__))
                message = 'Не удалось выполнить команду. Попробуй позже.'
            try:
                await reply(interaction, message, ephemeral=True)
            except discord.HTTPException:
                LOG.exception('Не удалось отправить сообщение об ошибке Discord.')

    async def ask_question(self, interaction: discord.Interaction, question: str) -> None:
        if not question.strip():
            await reply(interaction, 'Введи непустой вопрос.', ephemeral=True)
            return
        if self.groq is None:
            await reply(interaction, 'Нейросеть не подключена: нужен ключ Groq.', ephemeral=True)
            return
        if self.ask_slots.locked():
            await reply(interaction, 'Сейчас отвечаю на два вопроса. Попробуй чуть позже.', ephemeral=True)
            return
        async with self.ask_slots:
            await interaction.response.defer()
            try:
                async with asyncio.timeout(35):
                    completion = await self.groq.chat.completions.create(
                        model=self.config.groq_model,
                        messages=[{'role': 'user', 'content': question}],
                        temperature=0.7,
                        max_tokens=1024,
                    )
                answer = completion.choices[0].message.content if completion.choices else None
            except RateLimitError:
                LOG.warning('Достигнут лимит запросов Groq.')
                await reply(interaction, 'Лимит Groq исчерпан. Попробуй позже.')
                return
            except (APIError, TimeoutError):
                LOG.exception('Ошибка запроса к Groq.')
                await reply(interaction, 'Нейросеть сейчас недоступна. Попробуй позже.')
                return
        if not answer or not answer.strip():
            await reply(interaction, 'Нейросеть вернула пустой ответ. Попробуй переформулировать вопрос.')
            return
        # Discord failures are handled by the command error handler, not reported as Groq failures.
        await reply(interaction, f'**Вопрос:** {discord.utils.escape_markdown(question)}\n**Братка:** {answer}')


async def run_bot(config: Config) -> None:
    async with Bratka(config, StatsStore(ROOT / 'users.json')) as bot:
        loop = asyncio.get_running_loop()
        runner = asyncio.current_task()
        installed_signals = []
        for sig in (signal.SIGTERM,):
            try:
                loop.add_signal_handler(sig, runner.cancel)
                installed_signals.append(sig)
            except NotImplementedError:
                pass  # Windows: Ctrl+C is handled by asyncio.run.
        try:
            await bot.start(config.token)
        finally:
            for sig in installed_signals:
                loop.remove_signal_handler(sig)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    try:
        config = load_config(ROOT / 'config.json')
        with contextlib.suppress(KeyboardInterrupt, asyncio.CancelledError):
            asyncio.run(run_bot(config))
    except (ValueError, OSError, discord.LoginFailure):
        LOG.exception('Бот не запущен или не смог сохранить данные при остановке.')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
