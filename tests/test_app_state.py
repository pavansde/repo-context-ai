from pathlib import Path

from core.app_state import AppState
from core.config import get_settings


def test_app_state_startup_and_shutdown(
    tmp_path: Path,
):
    checkpoint_db = (
        tmp_path / "data" / "checkpoints.db"
    )

    settings = get_settings().model_copy(
        update={
            "checkpoint_db_path": checkpoint_db,
        }
    )

    state = AppState(settings)

    assert state.connection is None
    assert state.checkpointer is None

    state.startup()

    assert state.connection is not None
    assert state.checkpointer is not None
    assert checkpoint_db.exists()

    state.shutdown()

    assert state.connection is None
    assert state.checkpointer is None