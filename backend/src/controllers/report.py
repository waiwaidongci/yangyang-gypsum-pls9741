from fastapi.responses import Response

from src.services.reviews import review_info
from src.services.runs import (
    current_release,
    current_run_id_for_dataset,
    export_current_table,
    export_table,
    get_run,
    run_info,
)
from src.utils.csv_files import write_rows


def report_detail(session, run_id, actor):
    run, review = get_run(session, run_id, actor)
    current_run_id = current_run_id_for_dataset(session, run.dataset_id)
    return {**run_info(run, review, current_run_id), 'result': run.result, 'error': run.error,
            'current_run_id': current_run_id, 'active_run_id': current_run_id,
            'review': review_info(review) if review else None}


def report_csv(session, run_id, key, actor):
    table = export_table(session, run_id, key, actor)
    return Response(write_rows(table['columns'], table['rows']), media_type='text/csv',
                    headers={'Content-Disposition': f'attachment; filename="run-{run_id}-{table["key"]}.csv"'})


def current_report_csv(session, key, actor, dataset_id: int | None = None):
    table, run = export_current_table(session, key, dataset_id, actor)
    response = Response(
        write_rows(table['columns'], table['rows']),
        media_type='text/csv',
        headers={
            'Content-Disposition': f'attachment; filename="run-{run.id}-{table["key"]}.csv"',
            'X-Handoff-Run-Id': str(run.id),
        },
    )
    return response


def current_release_detail(session, dataset_id: int | None = None):
    run, review = current_release(session, dataset_id)
    return {
        'run_id': run.id,
        'current_run_id': run.id,
        'active_run_id': run.id,
        'id': run.id,
        'dataset_id': run.dataset_id,
        'decision': review.decision,
        'review': review_info(review),
        'download_base': f'/api/runs/{run.id}/export',
    }
