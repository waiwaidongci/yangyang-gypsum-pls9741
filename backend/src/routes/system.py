from fastapi import APIRouter, Query
from sqlalchemy import select, text

from src.controllers.dependencies import Actor, Admin, Db
from src.controllers.report import current_release_detail, current_report_csv
from src.domain.project import SPEC
from src.models.audit import Audit
from src.services.queue import redis_client

router = APIRouter()


@router.get('/health')
def health(session: Db):
    session.execute(text('SELECT 1'))
    redis_client.ping()
    return {'status': 'ok'}


@router.get('/spec')
def project_spec():
    return SPEC


@router.get('/downloads/{key}')
@router.get('/export/{key}')
@router.get('/exports/current/{key}')
@router.get('/current-downloads/{key}')
@router.get('/active-downloads/{key}')
def global_download(key: str, session: Db, actor: Actor, dataset_id: int | None = Query(default=None)):
    return current_report_csv(session, key, actor, dataset_id)


@router.get('/handoff/current')
def global_handoff_current(session: Db, actor: Actor, dataset_id: int | None = Query(default=None)):
    return current_release_detail(session, dataset_id)


@router.get('/handoff/downloads/{key}')
def handoff_download(key: str, session: Db, actor: Actor, dataset_id: int | None = Query(default=None)):
    return current_report_csv(session, key, actor, dataset_id)


@router.get('/handoff/{dataset_id}/{key}')
def handoff_dataset_table(dataset_id: int, key: str, session: Db, actor: Actor):
    return current_report_csv(session, key, actor, dataset_id)


@router.get('/handoffs/current')
def handoffs_current(session: Db, actor: Actor, dataset_id: int | None = Query(default=None)):
    return current_release_detail(session, dataset_id)


@router.get('/handoffs/current/export/{key}')
def handoffs_current_export(key: str, session: Db, actor: Actor, dataset_id: int | None = Query(default=None)):
    return current_report_csv(session, key, actor, dataset_id)


@router.get('/audit')
def events(session: Db, actor: Admin):
    fields = ('id', 'actor_id', 'action', 'object_id', 'created_at')
    return [{key: getattr(event, key) for key in fields} for event in session.scalars(
        select(Audit).order_by(Audit.id.desc()).limit(50))]
