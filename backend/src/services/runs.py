import math

from fastapi import HTTPException
from sqlalchemy import select

from src.constants.workflow import Decision, Role, RunStatus
from src.domain.project import SPEC
from src.models.review import Review
from src.models.run import Run
from src.services.audit import record
from src.services.catalog import get_dataset
from src.services.queue import notify_worker
from src.utils.time import utc_stamp


def run_info(run, review, current_run_id: int | None = None):
    fields = ('id', 'label', 'dataset_id', 'created_at', 'status', 'settings')
    decision = review.decision if review else None
    is_current = None if current_run_id is None else run.id == current_run_id
    return {
        **{key: getattr(run, key) for key in fields},
        'decision': decision,
        'current': is_current,
        'is_current': is_current,
        'active': is_current,
        'is_active': is_current,
        # Every approved run remains downloadable by its immutable run URL;
        # only one is marked as the stable/current handoff per dataset.
        'released': decision == Decision.APPROVED if current_run_id is not None else None,
    }


def current_run_id_map(session, dataset_ids=None) -> dict[int, int]:
    statement = select(Run.id, Run.dataset_id).join(Review, Review.run_id == Run.id).where(
        Review.decision == Decision.APPROVED,
    ).order_by(Review.created_at.desc(), Run.id.desc())
    if dataset_ids is not None:
        statement = statement.where(Run.dataset_id.in_(list(dataset_ids)))
    mapping: dict[int, int] = {}
    for run_id, dataset_id in session.execute(statement):
        mapping.setdefault(dataset_id, run_id)
    return mapping


def current_run_id_for_dataset(session, dataset_id: int) -> int | None:
    return current_run_id_map(session, [dataset_id]).get(dataset_id)


def list_runs(session, actor):
    statement = select(Run, Review).outerjoin(Review, Run.id == Review.run_id).order_by(Run.id.desc()).limit(100)
    if actor.role == Role.VIEWER:
        statement = statement.where(Review.decision == Decision.APPROVED)
    rows = list(session.execute(statement))
    current_ids = current_run_id_map(session, [run.dataset_id for run, _review in rows])
    return [run_info(run, review, current_ids.get(run.dataset_id)) for run, review in rows]


def get_run(session, run_id, actor):
    run = session.get(Run, run_id)
    review = session.get(Review, run_id)
    if run is None or (actor.role == Role.VIEWER and (not review or review.decision != Decision.APPROVED)):
        raise LookupError('找不到可查看的运行')
    return run, review


def validate_settings(settings):
    parameters = {item['key']: item for item in SPEC['settings']}
    if settings.keys() - parameters.keys():
        raise ValueError('存在本项目未定义的运行参数')
    checked = {}
    for key, parameter in parameters.items():
        value = settings.get(key, parameter['default'])
        if not math.isfinite(value) or not parameter['min'] <= value <= parameter['max']:
            raise ValueError(f"{parameter['label']}超出范围")
        if parameter['step'] == 1 and value != int(value):
            raise ValueError(f"{parameter['label']}必须是整数")
        checked[key] = int(value) if parameter['step'] == 1 else value
    return checked


def create_run(session, payload, actor):
    get_dataset(session, payload.dataset_id)
    run = Run(label=payload.label, dataset_id=payload.dataset_id, created_by=actor.id,
              created_at=utc_stamp(), status=RunStatus.QUEUED, settings=validate_settings(payload.settings))
    session.add(run)
    session.flush()
    record(session, 'run.queued', run.id, actor.id)
    session.commit()
    notify_worker()
    return run_info(run, None)


def export_table(session, run_id, key, actor):
    run, review = get_run(session, run_id, actor)
    if review is None or review.decision != Decision.APPROVED:
        raise HTTPException(409, '复核通过后可下载正式 CSV')
    return table_for_key(run, key)


def table_for_key(run: Run, key: str) -> dict:
    if not run.result:
        raise LookupError('找不到结果表')
    for table in run.result['tables']:
        if table['key'] == key:
            return table
    raise LookupError('找不到结果表')


def current_release(session, dataset_id: int | None = None) -> tuple[Run, Review]:
    """Return the atomically switched download version.

    Approval inserts the review in the same transaction that makes a run
    downloadable.  Failed or merely completed runs have no approved review
    and therefore never replace the previous release.
    """
    statement = select(Run, Review).join(Review, Review.run_id == Run.id).where(
        Review.decision == Decision.APPROVED,
    )
    if dataset_id is not None:
        statement = statement.where(Run.dataset_id == dataset_id)
    statement = statement.order_by(Review.created_at.desc(), Run.id.desc()).limit(1)
    row = session.execute(statement).first()
    if row is None:
        message = '该台账还没有获批并可下载的交接版本' if dataset_id is not None else '还没有获批并可下载的交接版本'
        raise HTTPException(409, message)
    return row[0], row[1]


def export_current_table(session, key: str, dataset_id: int | None = None, actor=None):
    run, _review = current_release(session, dataset_id)
    return table_for_key(run, key), run


# Naming aliases for callers/tests describing the switched handoff release.
get_current_release = current_release
get_active_release = current_release
current_handoff = current_release
def export_active_table(session, key: str, dataset_id: int | None = None, actor=None):
    return export_current_table(session, key, dataset_id, actor)


def publish_run(session, run_id: int):
    """Validate that an already-reviewed run can be the switched release."""
    run = session.get(Run, run_id)
    review = session.get(Review, run_id)
    if run is None or review is None or review.decision != Decision.APPROVED:
        raise HTTPException(409, '只有复核通过的交接才能切换下载版本')
    return run_info(run, review, run.id)


switch_downloads = publish_run
publish_handoff = publish_run
