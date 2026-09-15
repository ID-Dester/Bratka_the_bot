import asyncio
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

import discord
import httpx
from groq import APIConnectionError, RateLimitError

import main
from main import Bratka, Config, load_config, split_message
from storage import DataError, StatsStore


class Interaction:
    def __init__(self):
        self.sent = []
        self.done = False
        self.guild_id = 1
        self.user = NS(id=2)
        self.response = NS(is_done=lambda: self.done, send_message=self.send, defer=self.defer)
        self.followup = NS(send=self.send)

    async def defer(self):
        self.done = True

    async def send(self, content, **kwargs):
        assert len(content.encode('utf-16-le')) // 2 <= 2000
        assert not kwargs['allowed_mentions'].everyone
        assert not kwargs['allowed_mentions'].users
        assert not kwargs['allowed_mentions'].roles
        self.sent.append(content)
        self.done = True


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'users.json'
        self.store = StatsStore(self.path)

    def test_roundtrip_and_previous_backup(self):
        self.assertEqual(self.store.load(), {})
        self.store.save({'1': {'2': 10}})
        self.store.save({'1': {'2': 11}})
        self.assertEqual(self.store.load(), {'1': {'2': 11}})
        self.assertEqual(json.loads(self.store.backup.read_text()), {'1': {'2': 10}})

    def test_corrupt_file_is_never_overwritten(self):
        for raw in ['{"1":', '[]', '{"1": {"2": true}}', '{"1": {"2": -1}}', '{"bad": {}}']:
            with self.subTest(raw=raw):
                self.path.write_text(raw, encoding='utf-8')
                with self.assertRaises(DataError):
                    self.store.load()
                with self.assertRaises(DataError):
                    self.store.save({})
                self.assertEqual(self.path.read_text(), raw)

    def test_invalid_encoding_is_preserved(self):
        self.path.write_bytes(b'\xff\xfe\x80')
        with self.assertRaises(DataError):
            self.store.load()
        with self.assertRaises(DataError):
            self.store.save({})
        self.assertEqual(self.path.read_bytes(), b'\xff\xfe\x80')

    def test_interrupted_replace_preserves_original(self):
        self.store.save({'1': {'2': 10}})
        import os
        replace = os.replace

        def fail_primary(source, destination):
            if destination == self.path:
                raise OSError('simulated interrupted write')
            replace(source, destination)

        with patch('storage.os.replace', side_effect=fail_primary):
            with self.assertRaises(OSError):
                self.store.save({'1': {'2': 11}})
        self.assertEqual(self.store.load(), {'1': {'2': 10}})
        self.assertEqual(json.loads(self.store.backup.read_text()), {'1': {'2': 10}})

    def test_missing_primary_with_backup_requires_recovery(self):
        self.store.backup.write_text('{"1": {"2": 10}}')
        with self.assertRaises(DataError):
            self.store.load()

    def test_config_compatible_with_old_format(self):
        self.path.write_text('{"token": " test-token ", "groq_key": ""}')
        config = load_config(self.path)
        self.assertEqual(config.token, 'test-token')
        self.assertNotIn('test-token', repr(config))
        self.assertEqual(config.groq_model, 'llama-3.1-8b-instant')

    def test_invalid_config_fails_clearly(self):
        for raw in ['{', '[]', '{}', '{"token": 42}', '{"token":"x", "groq_key": []}']:
            with self.subTest(raw=raw):
                self.path.write_text(raw)
                with self.assertRaises(ValueError):
                    load_config(self.path)


class BotTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = Mock(spec=StatsStore)
        self.store.load.return_value = {}
        self.bot = Bratka(Config('test-token'), self.store)
        self.interaction = Interaction()

    async def asyncTearDown(self):
        await self.bot.close()

    def provider(self, answer='Ответ'):
        create = AsyncMock(return_value=NS(choices=[NS(message=NS(content=answer))]))
        self.bot.groq = NS(chat=NS(completions=NS(create=create)), close=AsyncMock())
        return create

    async def test_dm_and_bot_messages_ignored(self):
        await self.bot.on_message(NS(guild=None, author=NS(id=2, bot=False)))
        await self.bot.on_message(NS(guild=NS(id=1), author=NS(id=2, bot=True)))
        self.assertEqual(self.bot.stats, {})

    async def test_guild_counters_are_isolated(self):
        for guild in [1, 1, 3]:
            await self.bot.on_message(NS(guild=NS(id=guild), author=NS(id=2, bot=False)))
        self.assertEqual(self.bot.stats, {'1': {'2': 2}, '3': {'2': 1}})

    async def test_long_answer_and_question_fit_and_are_not_lost(self):
        create = self.provider('🙂' * 2200)
        await self.bot.ask_question(self.interaction, 'в' * 1000)
        self.assertGreater(len(self.interaction.sent), 1)
        self.assertIn('🙂' * 2200, ''.join(self.interaction.sent))
        create.assert_awaited_once()

    async def test_groq_does_not_block_ping(self):
        entered, release = asyncio.Event(), asyncio.Event()
        create = self.provider()

        async def slow_provider(**kwargs):
            entered.set()
            await release.wait()
            return NS(choices=[NS(message=NS(content='Готово'))])

        create.side_effect = slow_provider
        pending = asyncio.create_task(self.bot.ask_question(self.interaction, 'Вопрос'))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            ping = Interaction()
            await asyncio.wait_for(self.bot.tree.get_command('ping').callback(ping), 1)
            self.assertEqual(ping.sent, ['pong'])
            self.assertFalse(pending.done())
        finally:
            release.set()
            await pending

    async def test_busy_provider_does_not_queue_more_requests(self):
        create = self.provider()
        async with self.bot.ask_slots, self.bot.ask_slots:
            await self.bot.ask_question(self.interaction, 'Вопрос')
        create.assert_not_awaited()
        self.assertIn('два вопроса', self.interaction.sent[0])

    async def test_provider_failures_release_slot(self):
        request = httpx.Request('POST', 'https://api.groq.com/openai/v1/chat/completions')
        response = httpx.Response(429, request=request)
        errors = [TimeoutError(), APIConnectionError(request=request),
                  RateLimitError('limited', response=response, body=None)]
        for error in errors:
            with self.subTest(error=type(error).__name__):
                create = self.provider()
                create.side_effect = error
                interaction = Interaction()
                with self.assertLogs('main', level='WARNING'):
                    await self.bot.ask_question(interaction, 'Вопрос')
                self.assertEqual(len(interaction.sent), 1)
                self.assertFalse(self.bot.ask_slots.locked())

    async def test_empty_provider_result(self):
        self.provider(None)
        await self.bot.ask_question(self.interaction, 'Вопрос')
        self.assertIn('пустой ответ', self.interaction.sent[0])

    async def test_discord_error_is_not_misreported_as_groq(self):
        self.provider()
        self.interaction.followup.send = AsyncMock(side_effect=discord.HTTPException(
            NS(status=400, reason='Bad Request'), 'send failed'))
        with self.assertRaises(discord.HTTPException):
            await self.bot.ask_question(self.interaction, 'Вопрос')
        self.interaction.followup.send.assert_awaited_once()

    async def test_disabled_ai_and_empty_question(self):
        await self.bot.ask_question(self.interaction, 'Вопрос')
        self.assertIn('ключ Groq', self.interaction.sent[0])
        create = self.provider()
        await self.bot.ask_question(self.interaction, '   ')
        create.assert_not_awaited()

    async def test_choice_empty_and_long_unicode(self):
        command = self.bot.tree.get_command('difficult_choice')
        await command.callback(self.interaction, ', , ')
        self.assertIn('хотя бы один', self.interaction.sent[0])
        long_interaction = Interaction()
        await command.callback(long_interaction, '🙂' * 900)
        self.assertIn('🙂' * 900, ''.join(long_interaction.sent))

    async def test_empty_top(self):
        await self.bot.tree.get_command('top').callback(self.interaction)
        self.assertIn('нет статистики', self.interaction.sent[0])

    async def test_top_survives_missing_user(self):
        self.bot.stats = {'1': {'2': 7}}
        with patch.object(self.bot, 'fetch_user', new=AsyncMock(side_effect=discord.NotFound(
                NS(status=404, reason='Not Found'), 'unknown user'))):
            with self.assertLogs('main', level='WARNING'):
                await self.bot.tree.get_command('top').callback(self.interaction)
        self.assertIn('Пользователь 2 — 7', self.interaction.sent[0])

    async def test_close_flushes_and_closes_provider_even_on_save_error(self):
        self.provider()
        self.store.save.side_effect = OSError('disk full')
        with self.assertRaises(OSError):
            await self.bot.close()
        self.bot.groq.close.assert_awaited_once()
        self.assertTrue(self.bot.is_closed())

    async def test_close_flushes_latest_messages_and_stops_autosave(self):
        self.bot.autosave_task = asyncio.create_task(self.bot.autosave())
        await self.bot.on_message(NS(guild=NS(id=1), author=NS(id=2, bot=False)))
        await asyncio.wait_for(self.bot.close(), 1)
        self.store.save.assert_called_once_with({'1': {'2': 1}})
        self.assertTrue(self.bot.autosave_task.done())

    async def test_save_snapshot_does_not_lose_messages_arriving_during_write(self):
        import threading
        writing, release = threading.Event(), threading.Event()
        snapshots = []

        def slow_save(snapshot):
            writing.set()
            if not release.wait(timeout=2):
                raise TimeoutError('test writer was not released')
            snapshots.append(snapshot)

        self.store.save.side_effect = slow_save
        message = NS(guild=NS(id=1), author=NS(id=2, bot=False))
        await self.bot.on_message(message)
        pending = asyncio.create_task(self.bot.save_stats())
        try:
            self.assertTrue(await asyncio.to_thread(writing.wait, 1))
            await self.bot.on_message(message)
        finally:
            release.set()
            await pending
        self.assertEqual(snapshots[0], {'1': {'2': 1}})
        await self.bot.close()
        self.assertEqual(snapshots[-1], {'1': {'2': 2}})

    async def test_autosave_retries_after_disk_error(self):
        calls = 0

        async def tick(awaitable, timeout):
            nonlocal calls
            awaitable.close()
            calls += 1
            if calls <= 2:
                raise TimeoutError
            self.bot.stop_saving.set()

        with patch('main.asyncio.wait_for', side_effect=tick), \
                patch.object(self.bot, 'save_stats', new=AsyncMock(side_effect=[OSError('disk full'), None])) as save:
            with self.assertLogs('main', level='ERROR'):
                await self.bot.autosave()
            self.assertEqual(save.await_count, 2)

    async def test_only_required_intents_and_guild_only_top(self):
        self.assertTrue(self.bot.intents.guild_messages)
        self.assertFalse(self.bot.intents.message_content)
        self.assertFalse(self.bot.intents.members)
        self.assertFalse(self.bot.intents.presences)
        self.assertTrue(self.bot.tree.get_command('top').guild_only)


class PureTests(unittest.TestCase):
    def test_import_has_no_io_or_bot_start(self):
        with patch('discord.Client.start', side_effect=AssertionError('network')), \
                patch.object(Path, 'read_text', side_effect=AssertionError('configuration read')):
            importlib.reload(main)

    def test_message_split_preserves_text(self):
        for text in ['x' * 2024, '🙂' * 2500, 'первая\nвторая', '']:
            chunks = split_message(text)
            self.assertEqual(''.join(chunks), text)
            self.assertTrue(all(len(chunk.encode('utf-16-le')) // 2 <= 1900 for chunk in chunks))


if __name__ == '__main__':
    unittest.main()
