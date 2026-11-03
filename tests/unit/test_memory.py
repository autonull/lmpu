"""Tests for LMPU memory module."""

import pytest
import tempfile
import numpy as np

from lmpu.memory import (
    BufferManager,
    TempSHMBackend,
    GrantManager,
    GrantType,
    Grant,
    ChecksumAlgorithm,
    compute_checksum,
    verify_checksum,
)


class TestChecksum:
    """Test checksum utilities."""
    
    def test_blake3_checksum(self):
        data = b"hello world"
        checksum = compute_checksum(data, ChecksumAlgorithm.BLAKE3)
        assert checksum.startswith("blake3:")
        assert verify_checksum(data, checksum, ChecksumAlgorithm.BLAKE3)
    
    def test_sha256_checksum(self):
        data = b"hello world"
        checksum = compute_checksum(data, ChecksumAlgorithm.SHA256)
        assert checksum.startswith("sha256:")
        assert verify_checksum(data, checksum, ChecksumAlgorithm.SHA256)
    
    def test_verify_fails_on_wrong_data(self):
        data = b"hello world"
        checksum = compute_checksum(data, ChecksumAlgorithm.BLAKE3)
        assert not verify_checksum(b"different", checksum, ChecksumAlgorithm.BLAKE3)


class TestGrantManager:
    """Test grant management."""
    
    def test_grant_read(self):
        mgr = GrantManager()
        grant = mgr.grant("job_1", "buf_1", GrantType.READ, 0, 100)
        
        assert grant.job_id == "job_1"
        assert grant.buffer_id == "buf_1"
        assert grant.grant_type == GrantType.READ
        assert grant.can_read()
        assert not grant.can_write()
    
    def test_grant_write(self):
        mgr = GrantManager()
        grant = mgr.grant("job_1", "buf_1", GrantType.WRITE, 0, 100)
        
        assert grant.can_write()
        assert not grant.can_read()
    
    def test_grant_read_write(self):
        mgr = GrantManager()
        grant = mgr.grant("job_1", "buf_1", GrantType.READ_WRITE, 0, 100)
        
        assert grant.can_read()
        assert grant.can_write()
    
    def test_check_access(self):
        mgr = GrantManager()
        mgr.grant("job_1", "buf_1", GrantType.READ, 0, 100)
        
        # Should have access
        grant = mgr.check_access("job_1", "buf_1", 0, 50, write=False)
        assert grant is not None
        
        # Should not have write access
        grant = mgr.check_access("job_1", "buf_1", 0, 50, write=True)
        assert grant is None
        
        # Different job should not have access
        grant = mgr.check_access("job_2", "buf_1", 0, 50, write=False)
        assert grant is None
    
    def test_revoke_job(self):
        mgr = GrantManager()
        mgr.grant("job_1", "buf_1", GrantType.READ)
        mgr.grant("job_1", "buf_2", GrantType.WRITE)
        
        count = mgr.revoke_job("job_1")
        assert count == 2
        
        # No grants should remain
        grants = mgr.get_job_grants("job_1")
        assert len(grants) == 0


class TestBufferManager:
    """Test buffer manager with temp backend."""
    
    @pytest.fixture
    def buffer_manager(self):
        backend = TempSHMBackend()
        mgr = BufferManager(backend=backend)
        yield mgr
        backend.cleanup()
    
    def test_create_buffer(self, buffer_manager):
        ref = buffer_manager.create(
            session_id="session_1",
            name="test",
            dtype="float32",
            shape=(100, 100),
        )
        
        assert ref.buffer_id.startswith("test_")
        assert ref.dtype == "float32"
        assert ref.shape == [100, 100]
        assert ref.bytes == 100 * 100 * 4  # float32 = 4 bytes
    
    def test_import_bytes(self, buffer_manager):
        data = np.random.randn(50, 50).astype(np.float32).tobytes()
        ref = buffer_manager.import_bytes(
            session_id="session_1",
            name="imported",
            data=data,
            dtype="float32",
            shape=(50, 50),
        )
        
        assert ref.bytes == len(data)
        assert ref.checksum != ""
    
    def test_import_file(self, buffer_manager):
        import os
        # Create a temp file in the import roots directory
        import_root = buffer_manager.import_roots[0]
        os.makedirs(import_root, exist_ok=True)
        
        temp_path = os.path.join(import_root, "test_import.bin")
        data = np.random.randn(20, 20).astype(np.float32)
        
        with open(temp_path, "wb") as f:
            f.write(data.tobytes())
        
        try:
            ref = buffer_manager.import_file(
                session_id="session_1",
                name="from_file",
                path=temp_path,
                dtype="float32",
                shape=(20, 20),
            )
            
            assert ref.bytes == 20 * 20 * 4
        finally:
            os.unlink(temp_path)
    
    def test_stat(self, buffer_manager):
        ref = buffer_manager.create(
            session_id="session_1",
            name="stat_test",
            dtype="int32",
            shape=(10, 10),
        )
        
        stat = buffer_manager.stat(ref.buffer_id)
        assert stat.buffer_id == ref.buffer_id
        assert stat.dtype == "int32"
        assert stat.shape == [10, 10]
    
    def test_checksum(self, buffer_manager):
        ref = buffer_manager.create(
            session_id="session_1",
            name="checksum_test",
            dtype="float64",
            shape=(5, 5),
        )
        
        checksum = buffer_manager.checksum(ref.buffer_id)
        assert checksum.startswith("blake3:")
    
    def test_delete(self, buffer_manager):
        ref = buffer_manager.create(
            session_id="session_1",
            name="delete_test",
            dtype="float32",
            shape=(10, 10),
        )
        
        deleted = buffer_manager.delete(ref.buffer_id)
        assert deleted is True
        
        # Should not exist anymore
        with pytest.raises(Exception):
            buffer_manager.stat(ref.buffer_id)
    
    def test_grant_read_write(self, buffer_manager):
        ref = buffer_manager.create(
            session_id="session_1",
            name="grant_test",
            dtype="float32",
            shape=(100, 100),
        )
        
        # Grant read
        grant = buffer_manager.grant_read("job_1", ref.buffer_id)
        assert grant.grant_type == GrantType.READ
        
        # Grant write
        grant = buffer_manager.grant_write("job_1", ref.buffer_id)
        assert grant.grant_type == GrantType.WRITE
        
        # Grant read_write
        grant = buffer_manager.grant_read_write("job_1", ref.buffer_id)
        assert grant.grant_type == GrantType.READ_WRITE
    
    def test_release_grants(self, buffer_manager):
        ref = buffer_manager.create(
            session_id="session_1",
            name="release_test",
            dtype="float32",
            shape=(100, 100),
        )
        
        buffer_manager.grant_read("job_1", ref.buffer_id)
        buffer_manager.grant_write("job_1", ref.buffer_id)
        
        count = buffer_manager.release_grants("job_1")
        assert count == 2