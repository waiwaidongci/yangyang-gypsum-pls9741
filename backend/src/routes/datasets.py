from fastapi import APIRouter, Query

from src.controllers.dependencies import Actor, Db, Writer
from src.controllers.report import current_release_detail, current_report_csv
from src.schemas.datasets import DatasetImport, DatasetInfo
from src.services.catalog import get_dataset, import_dataset, list_datasets, overview
from src.services.runs import current_run_id_for_dataset

router = APIRouter(prefix='/datasets')


@router.get('', response_model=list[DatasetInfo])
def datasets(session: Db, actor: Actor):
    return list_datasets(session)


@router.get('/{dataset_id}')
def detail(dataset_id: int, session: Db, actor: Actor, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
    dataset = get_dataset(session, dataset_id)
    current_run_id = current_run_id_for_dataset(session, dataset_id)
    return {
        **overview(dataset),
        'current_run_id': current_run_id,
        'active_run_id': current_run_id,
        'rows': dataset.rows[offset:offset + limit],
    }


@router.get('/{dataset_id}/runs/current')
@router.get('/{dataset_id}/runs/latest')
@router.get('/{dataset_id}/current')
@router.get('/{dataset_id}/active')
def current(dataset_id: int, session: Db, actor: Actor):
    get_dataset(session, dataset_id)
    return current_release_detail(session, dataset_id)


@router.get('/{dataset_id}/downloads/{key}')
@router.get('/{dataset_id}/download/{key}')
@router.get('/{dataset_id}/export/{key}')
@router.get('/{dataset_id}/exports/{key}')
@router.get('/{dataset_id}/handoff/{key}')
def download_current(dataset_id: int, key: str, session: Db, actor: Actor):
    get_dataset(session, dataset_id)
    return current_report_csv(session, key, actor, dataset_id)


@router.get('/{dataset_id}/runs/current/export/{key}')
@router.get('/{dataset_id}/runs/latest/export/{key}')
@router.get('/{dataset_id}/current/export/{key}')
def current_export_nested(dataset_id: int, key: str, session: Db, actor: Actor):
    get_dataset(session, dataset_id)
    return current_report_csv(session, key, actor, dataset_id)


@router.get('/{dataset_id}/active-downloads/{key}')
def active_download_current(dataset_id: int, key: str, session: Db, actor: Actor):
    get_dataset(session, dataset_id)
    return current_report_csv(session, key, actor, dataset_id)


@router.post('', response_model=DatasetInfo, status_code=201)
def insert(payload: DatasetImport, session: Db, actor: Writer):
    return import_dataset(session, payload, actor.id)
