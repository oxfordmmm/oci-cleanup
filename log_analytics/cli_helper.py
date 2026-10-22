from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from threading import Lock

from oci.log_analytics import LogAnalyticsClient
from oci.log_analytics.models import LogAnalyticsEntitySummary
from tqdm import tqdm

from log_analytics.sdk import (
    delete_log_analytics_entity_with_retry,
    list_log_analytics_entities,
)


def clean_log_analytics_entities(
    log_analytics_client: LogAnalyticsClient,
    compartment_id: str,
    namespace: str,
    num_workers=1,
):
    """Delete all log analytics entities in a compartment"""
    # Get list of log analytics entities
    entities: list[LogAnalyticsEntitySummary] = list_log_analytics_entities(
        log_analytics_client, compartment_id, namespace
    )

    if not entities:
        print(f"No log analytics entities found in compartment '{compartment_id}'")
        return

    print(f"\nFound {len(entities)} log analytics entities to delete")

    # Create queue and progress bar for entity deletion
    entity_queue: Queue[LogAnalyticsEntitySummary] = Queue()
    progress_lock: Lock = Lock()
    print(f"Running with {num_workers} workers")

    # Create progress bar with percentage format
    with tqdm(
        total=len(entities),
        desc="Deleting log analytics entities",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        leave=False,
    ) as entity_pbar:
        # Fill the queue with entities
        for entity in entities:
            entity_queue.put(entity)

        # Create and start worker threads
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit worker tasks
            futures = [
                executor.submit(
                    delete_log_analytics_entity_worker,
                    log_analytics_client,
                    namespace,
                    entity_queue,
                    progress_lock,
                    entity_pbar,
                )
                for _ in range(num_workers)
            ]

            # Wait for all tasks to complete
            for future in futures:
                future.result()

            # Wait for queue to be empty
            entity_queue.join()


def delete_log_analytics_entity_worker(
    log_analytics_client: LogAnalyticsClient,
    namespace: str,
    queue: Queue[LogAnalyticsEntitySummary],
    progress_lock: Lock,
    entity_pbar,
):
    """Worker function to delete log analytics entities from the queue"""
    while True:
        try:
            entity: LogAnalyticsEntitySummary = queue.get_nowait()
            entity_id = entity.id
            entity_name = entity.name

            try:
                delete_log_analytics_entity_with_retry(
                    log_analytics_client, namespace, entity_id
                )
                with progress_lock:
                    percentage = (entity_pbar.n + 1) / entity_pbar.total * 100
                    entity_pbar.set_postfix_str(
                        f"[{percentage:.1f}%] Deleted: {entity_name}"
                    )
                    entity_pbar.update(1)
            except Exception as e:
                print(f"\nError deleting entity {entity_name} | ID: {entity_id}: {e}")
                with progress_lock:
                    entity_pbar.update(1)
            finally:
                queue.task_done()

        except Empty:
            break
