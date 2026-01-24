from __future__ import annotations

import msgpack
import os
import shutil
import unittest
import uuid
from collections import deque

from snake import config
from snake.memory import SymbolicMemory


class MemoryPersistenceTest(unittest.TestCase):
    def setUp(self):
        self._original_state_dir = config.STATE_DIR
        self._tmpdir = os.path.abspath(
            os.path.join(os.getcwd(), f"tmp-memory-{uuid.uuid4().hex}")
        )
        os.makedirs(self._tmpdir, exist_ok=True)
        config.set_state_dir(self._tmpdir)

    def tearDown(self):
        config.set_state_dir(self._original_state_dir)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_memory_roundtrip_preserves_state(self):
        memory = SymbolicMemory()
        fake_state = {
            "snake_body": deque([(0, 0), (0, 1)]),
            "food": (2, 2),
            "direction": (0, 1),
            "snake_head": (0, 0),
            "distance_sq": 8,
            "safe_moves": [(0, 1)],
        }
        memory.update_memory(fake_state, (0, 1), 3.0)
        memory.save_memory()

        reloaded = SymbolicMemory()
        self.assertFalse(reloaded.is_modified)
        self.assertEqual(len(reloaded.memory), len(memory.memory))

    def test_legacy_payload_triggers_rewrite_flag(self):
        legacy_state = {
            (0, 0, 0): {
                "state_info": {},
                "actions": {},
                "visits": 0,
                "last_visit_step": 0,
            }
        }
        with open(config.MEMORY_FILE, "wb") as fh:
            fh.write(msgpack.packb(legacy_state, use_bin_type=True))

        reloaded = SymbolicMemory()
        self.assertTrue(reloaded.is_modified)


class MemoryPruneTest(unittest.TestCase):
    def setUp(self):
        self.orig_max = config.MEMORY_MAX_ENTRIES
        self.orig_recency = config.MEMORY_RECENCY_WEIGHT
        config.MEMORY_MAX_ENTRIES = 2
        config.MEMORY_RECENCY_WEIGHT = 0.002

    def tearDown(self):
        config.MEMORY_MAX_ENTRIES = self.orig_max
        config.MEMORY_RECENCY_WEIGHT = self.orig_recency

    def test_prune_memory_trims_entries(self):
        memory = SymbolicMemory()
        memory.memory = {
            ((i,),): {
                "visits": i + 1,
                "last_visit_step": memory.total_updates - i,
                "actions": {},
            }
            for i in range(5)
        }
        memory.total_updates = 100
        memory.prune_memory()
        self.assertLessEqual(len(memory.memory), config.MEMORY_MAX_ENTRIES)
